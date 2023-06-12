#pragma once

#include <vector>
#include <thread>

#include "GL/glew.h"
#include <glm/glm.hpp>

#include "Particle.h"
#include "Geometry.h"

#define THREAD_COUNT 8

class SPHSystem
{
private:
	//particle data
	
	std::vector<std::vector<Particle*>> neighbouringParticles;
	bool started;

	//initializes the particles that will be used
	void initParticles();

	// Creates hash table for particles in infinite domain
	void buildTable();

	// Sphere geometry for rendering
	Geometry* sphere;
	glm::mat4 sphereScale;
	glm::mat4* sphereModelMtxs;
	GLuint vbo;

	// Threads and thread blocks
	std::thread threads[THREAD_COUNT];
	int blockBoundaries[THREAD_COUNT + 1];
	int tableBlockBoundaries[THREAD_COUNT + 1];

public:
	SPHSystem(unsigned int numParticles, float mass1, float mass2, float restDensity1, float restDensity2, float gasConst, float viscosity1, float viscosity2, float h, float g, float tension);
	~SPHSystem();

	//kernel/fluid constants
	float POLY6, POLY6_GRAD, POLY6_LAP, SPIKY_GRAD, SPIKY_LAP, GAS_CONSTANT, MASS1,MASS2, H2, SELF_DENS1, SELF_DENS2;
	unsigned int numParticles;
	//fluid properties
	float restDensity1;
	float restDensity2;
	float viscosity1, viscosity2, h, g, tension;

	std::vector<Particle*> particles;
	Particle** particleTable;
	glm::ivec3 getCell(Particle *p) const;

	// std::mutex mtx;

	//updates the SPH system
	void update(float deltaTime);

	//draws the SPH system & its particles
	void draw(const glm::mat4& viewProjMtx, GLuint shader);

	void reset();
	void startSimulation();
	void pause();
	void single();
};

