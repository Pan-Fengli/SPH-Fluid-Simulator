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
	int type;//���ֲ�ͬ���������ͣ�����ֻ�����ࣩ1,2
	float ci;//color���ԣ�ci=0.5,-0.5;
	float gas_constant;
	float viscosity;

	Particle(float mass, float size, glm::vec3 position, glm::vec3 velocity,int type,float ci);
	~Particle();
};

