/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>
#include <map>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::map;
using std::normal_distribution;
using std::discrete_distribution;
using std::default_random_engine;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO/DONE: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO/DONE: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  num_particles = 64;  // TODO/DONE: Set the number of particles

  default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  weights = vector<double>(num_particles, 1);

  for (int i = 0; i < num_particles; i++) {
    double sample_x = dist_x(gen);
    double sample_y = dist_y(gen);
    double sample_theta = dist_theta(gen); 

    Particle p;

    p.id = i;
    p.x = sample_x;
    p.y = sample_y;
    p.theta = sample_theta;
    p.weight = 1;

    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO/DONE: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  default_random_engine gen;

  for (auto& p: particles) {
    double x_f;
    double y_f;
    double theta_f;

    if (are_equal(yaw_rate, 0)) {
      double distance = velocity * delta_t;
      theta_f = p.theta;
      x_f = p.x + distance * cos(p.theta);
      y_f = p.y + distance * sin(p.theta);
    } else {
      double v_to_yaw = velocity / yaw_rate;
      theta_f = p.theta + yaw_rate * delta_t;
      x_f = p.x + v_to_yaw * (sin(theta_f) - sin(p.theta));
      y_f = p.y + v_to_yaw * (cos(p.theta) - cos(theta_f));
    }

    normal_distribution<double> dist_x(x_f, std_pos[0]);
    normal_distribution<double> dist_y(y_f, std_pos[1]);
    normal_distribution<double> dist_theta(theta_f, std_pos[2]);

    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO/DONE: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (auto& o: observations) {
    double min_dist = -1;
    int min_id = -1;
    for (auto& p: predicted) {
      double d = dist(o.x, o.y, p.x, p.y);
      if (min_dist < 0 || d < min_dist) {
        min_dist = d;
        min_id = p.id;
      }

      o.id = min_id;
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO/DONE: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  map<int, Map::single_landmark_s> id_to_landmark;
  for (auto& lm : map_landmarks.landmark_list) {
    id_to_landmark[lm.id_i] = lm;
  }

  for (int i = 0; i < particles.size(); i++) {
    Particle& p = particles[i];
    vector<LandmarkObs> observations_map = toMapCoordinates(p, observations);
    vector<LandmarkObs> predictions = predictLandmarks(p, map_landmarks, sensor_range);
    dataAssociation(predictions, observations_map);
    double weight = calculateWeight(observations_map, id_to_landmark, std_landmark[0], std_landmark[1]);

    p.weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO/DONE: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  default_random_engine gen;
  discrete_distribution<int> distribution(weights.begin(), weights.end());
  vector<Particle> resampled_particles;
  
  for (int i = 0; i < num_particles; i++) {
    int idx = distribution(gen);
    Particle p = particles[idx];
    resampled_particles.push_back(p);
  }
  
  particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

vector<LandmarkObs> ParticleFilter::toMapCoordinates(const Particle& p, vector<LandmarkObs> observations) {
  vector<LandmarkObs> observations_map;

  for (auto& o : observations) {
      LandmarkObs o_map;
      o_map.id = o.id;
      o_map.x = o.x * cos(p.theta) - o.y * sin(p.theta) + p.x;
      o_map.y = o.x * sin(p.theta) + o.y * cos(p.theta) + p.y;
      observations_map.push_back(o_map);
  }

  return observations_map;
}

vector<LandmarkObs> ParticleFilter::predictLandmarks(const Particle& p, Map map_landmarks, double sensor_range) {
 vector<LandmarkObs> predictions;

  for (auto& lm : map_landmarks.landmark_list) {
    if (dist(lm.x_f, lm.y_f, p.x, p.y) <= sensor_range) {
      LandmarkObs prediction; 
      prediction.id = lm.id_i;
      prediction.x = lm.x_f;
      prediction.y = lm.y_f;
      predictions.push_back(prediction);
    }
  }

  return predictions;
}

double ParticleFilter::calculateWeight(
  const vector<LandmarkObs>& observations,
  map<int, Map::single_landmark_s>& id_to_landmark,
  double std_x, double std_y
) {
  double weight = 1;
  for (auto& o : observations) {
    Map::single_landmark_s predicted_lm = id_to_landmark[o.id];
    weight *= multiv_prob(std_x, std_y, o.x, o.y, predicted_lm.x_f, predicted_lm.y_f);
  }
  return weight;
}
