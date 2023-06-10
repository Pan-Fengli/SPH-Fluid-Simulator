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
	float mass, size, elasticity;
	glm::vec3 position, velocity, acceleration;
	glm::vec3 force;

	// For linked list
	Particle* next;

	float density;
	float pressure;
	float id;
	int type;//区分不同的粒子类型（这里只有两类）1,2
	float ci;//color属性，ci=0.5,-0.5;
	float gas_constant;
	float viscosity;

	Particle(float mass, float size, glm::vec3 position, glm::vec3 velocity,int type,float ci);
	~Particle();
};

