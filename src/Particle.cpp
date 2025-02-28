#include "Particle.h"
#include <iostream>
#include <glm/gtx/string_cast.hpp>

int Particle::pCount = 0;

Particle::Particle(double mass, float size, glm::dvec3 position, glm::dvec3 velocity, int type, float ci) {
	this->mass = mass;
	this->size = size;
	this->position = position;
	this->velocity = velocity;
	this->type = type;
	this->ci = ci;

	force = glm::vec3(0);
	acceleration = glm::vec3(0);
	particle_velocity_new = glm::vec3(0);
	particle_pressure_acc = glm::vec3(0);

	density = 0.0f;
	pressure = 0.0f;
	id = pCount++;

	next = NULL;
}

Particle::~Particle() {}