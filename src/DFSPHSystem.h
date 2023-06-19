#pragma once

#include <vector>
#include <thread>

#include "GL/glew.h"
#include <glm/glm.hpp>

#include "Particle.h"
#include "Geometry.h"

#define THREAD_COUNT 8

class DFSPHSystem
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
	DFSPHSystem(unsigned int numParticles, float mass1, float mass2, double restDensity1, double restDensity2, float gasConst, double viscosity1, double viscosity2, double h, double g, double tension);
	~DFSPHSystem();

	//kernel/fluid constants
	double POLY6, POLY6_GRAD, POLY6_LAP, SPIKY_GRAD, SPIKY_LAP, GAS_CONSTANT, MASS1, MASS2, H2, SELF_DENS1, SELF_DENS2;
	double DeltaTime;
	double timeStep;
	double CFL_v = 0.30;// # CFL coefficient for velocity
	double CFL_a = 0.05;// # CFL coefficient for acceleration
	unsigned int numParticles;
	//fluid properties
	double restDensity1;
	double restDensity2;
	double viscosity1, viscosity2, h, dx, g, tension;
	double c_0 = 200;

	std::vector<Particle*> particles;
	Particle** particleTable;
	glm::ivec3 getCell(Particle* p) const;

	// std::mutex mtx;

	//updates the SPH system
	//void update(float deltaTime);
	void update1(float deltaTime);

	//draws the SPH system & its particles
	void draw(const glm::mat4& viewProjMtx, GLuint shader);

	void reset();
	void startSimulation();
	void pause();
	void single();
	float updateTimeStepSizeCFL();
	glm::vec3 CubicKernelGradW(glm::vec3& r) const;
	double cubic_kernel_derivative(double r_mod, double h) const;
	double DFSPHSystem::cubic_kernel(double r_mod, double h)const;
	void divergenceSolve();
	void densitySolver();
};

