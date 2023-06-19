#pragma once
#include "GL/glew.h"
#include <glm/glm.hpp>

class Particle
{
private:
	//for creating ids
	static int pCount;
public:
	// Attributes of particle
	double mass, size, elasticity;
	glm::dvec3 position, velocity, acceleration, particle_velocity_new, particle_pressure_acc;
	glm::dvec3 force;

	// For linked list
	Particle* next;

	double density;
	double pressure;
	float id;
	int type;//区分不同的粒子类型（这里只有两类）1,2
	float ci;//color属性，ci=0.5,-0.5;
	double gas_constant;
	double viscosity;
	double alpha;//DFSPH factor
	double densityAdv;
	double kai;
	double rho_predict;
	double particle_stiff;
	double d_density;//drho/dt;

	Particle(double mass, float size, glm::dvec3 position, glm::dvec3 velocity,int type,float ci);
	~Particle();
};

