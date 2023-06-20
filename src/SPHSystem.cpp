#include "SPHSystem.h"

#include <iostream>
#include<fstream>
using namespace std;
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/norm.hpp>

#define PI 3.14159265f
#define TABLE_SIZE 1000000

// This will lock across all instances of SPHSystem's,
// however since we only really have one instance, this
// should be okay for now  
std::mutex mtx;

/**
 * Hashes the position of a cell, giving where
 * the cell goes in the particle table 
 */
uint getHash(const glm::ivec3& cell) {
	return (
		(uint)(cell.x * 73856093) 
	  ^ (uint)(cell.y * 19349663) 
	  ^ (uint)(cell.z * 83492791)
	) % TABLE_SIZE;
}

SPHSystem::SPHSystem(unsigned int numParticles, float mass1, float mass2, float restDensity1, float restDensity2, float gasConst, float viscosity1, float viscosity2, float h, float g, float tension) {
	this->numParticles = numParticles;
	this->restDensity1 = restDensity1;
	this->restDensity2 = restDensity2;
	this->viscosity1 = viscosity1;
	this->viscosity2 = viscosity2;
	this->h = h;
	this->g = g;
	this->tension = tension;

	POLY6 = 315.0f / (64.0f * PI * pow(h, 9));
	SPIKY_GRAD = -45.0f / (PI * pow(h, 6));
	SPIKY_LAP = 45.0f / (PI * pow(h, 6));
	POLY6_GRAD = -945.0f/(32.0f * PI * pow(h, 9));
	POLY6_LAP = 945.0f / (32.0f * PI * pow(h, 9));
	MASS1 = mass1;
	MASS2 = mass2;
	GAS_CONSTANT = gasConst;
	H2 = h * h;
	SELF_DENS1 = MASS1 * POLY6 * pow(h, 6);
	SELF_DENS2 = MASS2 * POLY6 * pow(h, 6);

	DeltaTime = 0.001f;
	timeStep = DeltaTime;
	//setup densities & volume
	int cbNumParticles = numParticles * numParticles * numParticles;
	neighbouringParticles.resize(cbNumParticles);
	particles.resize(cbNumParticles);

	//initialize particles
	initParticles();

	// Load in sphere geometry and allocate matrice space
	sphere = new Geometry("resources/lowsphere.obj");
	sphereScale = glm::scale(glm::vec3(h/2.f));
	sphereModelMtxs = new glm::mat4[cbNumParticles];
	
	// Generate VBO for sphere model matrices
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(glm::mat4), &sphereModelMtxs[0], GL_DYNAMIC_DRAW);

	// Setup instance VAO
	glBindVertexArray(sphere->vao);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), 0);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)sizeof(glm::vec4));
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(sizeof(glm::vec4)*2));
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(sizeof(glm::vec4)*3));

	glVertexAttribDivisor(2,1);
	glVertexAttribDivisor(3,1);
	glVertexAttribDivisor(4,1);
	glVertexAttribDivisor(5,1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);


	

	//start init
	started = false;

	// Allocate table memory
	particleTable = (Particle **)malloc(sizeof(Particle *) * TABLE_SIZE);

	// Init block boundaries (for funcs that loop through particles)
	blockBoundaries[0] = 0;
	int blockSize = particles.size() / THREAD_COUNT;
	for (int i = 1; i < THREAD_COUNT; i++) {
		blockBoundaries[i] = i * blockSize;
	}
	blockBoundaries[THREAD_COUNT] = particles.size();

	// Init table block boundaries (for table clearing func)
	tableBlockBoundaries[0] = 0;
	blockSize = TABLE_SIZE / THREAD_COUNT;
	for (int i = 1; i < THREAD_COUNT; i++) {
		tableBlockBoundaries[i] = i * blockSize;
	}
	tableBlockBoundaries[THREAD_COUNT] = TABLE_SIZE;
}

SPHSystem::~SPHSystem() {
	// free table
	free(particleTable);
	free(sphereModelMtxs);

	//delete particles
	particles.clear();
	particles.shrink_to_fit();

	//delete neighbouring particles
	neighbouringParticles.clear();
	neighbouringParticles.shrink_to_fit();
}

void SPHSystem::initParticles() {
	std::srand(1024);
	float particleSeperation = h + 0.01f;
	int pcount = 0;
	int size = particles.size();
	for (int i = 0; i < numParticles; i++) {
		for (int j = 0; j < numParticles; j++) {
			for (int k = 0; k < numParticles; k++) {
				// dam like particle positions
				float ranX = (float(rand()) / float((RAND_MAX)) * 0.5f - 1) * h / 10;
				float ranY = (float(rand()) / float((RAND_MAX)) * 0.5f - 1) * h / 10;
				float ranZ = (float(rand()) / float((RAND_MAX)) * 0.5f - 1) * h / 10;
				glm::vec3 nParticlePos = glm::vec3(i * particleSeperation + ranX - 1.5f, j * particleSeperation + ranY + h + 0.1f, k * particleSeperation + ranZ - 1.5f);

				//create new particle
				//Particle* nParticle = new Particle(MASS, h,	nParticlePos, glm::vec3(0));
				Particle* nParticle;
				//初始化两种粒子
				if (pcount < size/2)
				{
					//红色的粒子
					nParticlePos.y += 0.1f;
					//nParticlePos.y -= 0.05f;
					nParticlePos.x -= particleSeperation;
					nParticle = new Particle(MASS1, h, nParticlePos, glm::vec3(0), 1, 0.5);
					nParticle->viscosity = viscosity1;
				}
				else {
					//蓝色的粒子
					nParticlePos.y += 0.1f;
					//nParticlePos.y -= 0.05f;
					nParticlePos.x += particleSeperation;
					nParticle = new Particle(MASS2, h, nParticlePos, glm::vec3(0), 2, -0.5);
					nParticle->viscosity = viscosity2;
				}

				//append particle
				particles[i + (j + numParticles * k) * numParticles] = nParticle;
				pcount++;
			}
		}
	}
}

/**
 * Parallel computation function for calculating density
 * and pressures of particles in the given SPH System.
 */
void parallelDensityAndPressures(const SPHSystem& sphSystem, int start, int end) {
	float massPoly6Product = sphSystem.MASS1 * sphSystem.POLY6;
	
	for (int i = start; i < end; i++) {
		float pDensity = 0;
		Particle* pi = sphSystem.particles[i];
		glm::ivec3 cell = sphSystem.getCell(pi);

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
					uint index = getHash(near_cell);
					Particle* pj = sphSystem.particleTable[index];
					
					// Iterate through cell linked list
					while (pj != NULL) {
						float dist2 = glm::length2(pj->position - pi->position);
						if (dist2 < sphSystem.H2 && pi != pj) {
							pDensity += pj->mass * sphSystem.POLY6 * glm::pow(sphSystem.H2 - dist2, 3);
						}
						pj = pj->next;
					}
				}
			}
		}
		
		// Include self density (as itself isn't included in neighbour)
		if (pi->type == 1)
		{
			pi->density = pDensity + sphSystem.SELF_DENS1;
		}
		else {
			pi->density = pDensity + sphSystem.SELF_DENS2;
		}
		

		// Calculate pressure
		float Rdens = pi->type == 1 ? sphSystem.restDensity1 : sphSystem.restDensity2;
		//float pPressure = sphSystem.GAS_CONSTANT * max(0.f,(pi->density - Rdens));//不允许是负的，不能够有收缩力?min?max?
		float pPressure = sphSystem.GAS_CONSTANT * (pi->density - Rdens);//不允许是负的，不能够有收缩力?min?max?
		if (pPressure < 0)
		{
			//printf("%f\n", pi->density);
			pPressure *= 0.00;
		}
		pi->pressure = pPressure;
	}
}

/**
 * Parallel computation function for calculating forces
 * of particles in the given SPH System.
 */
void parallelForces(const SPHSystem& sphSystem, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* pi = sphSystem.particles[i];
		pi->force = glm::vec3(0);
		glm::ivec3 cell = sphSystem.getCell(pi);

		glm::vec3 n= glm::vec3(0);//用于计算surface force的法向
		float lapCi=0;//Ci的拉普拉斯

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
					uint index = getHash(near_cell);
					Particle* pj = sphSystem.particleTable[index];

					// Iterate through cell linked list
					while (pj != NULL) {
						float dist2 = glm::length2(pj->position - pi->position);
						if (dist2 < sphSystem.H2 && pi != pj) {
							//unit direction and length
							float dist = sqrt(dist2);
							glm::vec3 dir = glm::normalize(pj->position - pi->position);

							//apply pressure force
							glm::vec3 pressureForce = -dir * (float)pj->mass * (float)(pi->pressure + pj->pressure) / (2 * (float)pj->density) * sphSystem.SPIKY_GRAD * (-1.0f);
							pressureForce *= std::pow(sphSystem.h - dist, 2);
							pi->force += pressureForce;

							//apply viscosity force
							glm::vec3 velocityDif = pj->velocity - pi->velocity;
							glm::vec3 viscoForce = (float)(pi->viscosity+pj->viscosity)/2 * (float)pj->mass * (velocityDif / (float)pj->density) * sphSystem.SPIKY_LAP * (sphSystem.h - dist);
							pi->force += viscoForce;

							//计算ci的梯度，以确定n
							glm::vec3 nci = dir * (float)pj->mass * (pj->ci - pi->ci) / (float)pj->density * sphSystem.POLY6_GRAD;//计算norm的时候用差值
							nci *= glm::pow(sphSystem.H2 - dist2, 2) * dist;
							n += nci;

							//计算ci的拉普拉斯
							float lci = pj->mass * (pj->ci - pi->ci) / pj->density * sphSystem.POLY6_LAP * (sphSystem.H2 - dist2) * (5 * dist2 - sphSystem.H2);
							lapCi += lci;
						}
						pj = pj->next;
					}
				}
			}
		}
		//计算完之后再加上边界力
		if (dot(n,n) > 0)
		{
			glm::normalize(n);
		}
		else {
			n = glm::vec3(0);
		}
		glm::vec3 intefaceForce = sphSystem.tension * lapCi * n;
		//printf("lapc:%f", lapCi);//0.00001
		if (i == 100)
		{
			//printf("before:%f,%f,%f \n", intefaceForce.x, intefaceForce.y, intefaceForce.z);
		}
		pi->force += intefaceForce;
		if (i == 100)
		{
			//printf("after:%f,%f,%f \n", pi->force.x, pi->force.y, pi->force.z);
		}
		//f.close();
	}
}

/**
 * Parallel computation function moving positions
 * of particles in the given SPH System.
 */
void parallelUpdateParticlePositions(const SPHSystem& sphSystem, float deltaTime, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle *p = sphSystem.particles[i];

		//calculate acceleration and velocity
		glm::dvec3 acceleration = p->force / p->density + glm::dvec3(0, sphSystem.g, 0);//重力怎么算？
		p->velocity += acceleration * (double)deltaTime;
		
		// Update position
		p->position += p->velocity * (double)deltaTime;

		// Handle collisions with box
		float boxWidth = 0.8f;
		float elasticity = 0.1f;
		float center = -1.5f + (sphSystem.numParticles * sphSystem.h)/2;
		if (p->position.y < p->size) {
			p->position.y = -p->position.y + 2 * p->size + 0.0001f;
			p->velocity.y = -p->velocity.y * elasticity;
		}

		if (p->position.x < p->size - boxWidth + center) {
			p->position.x = -p->position.x + 2 * (p->size - boxWidth + center) + 0.0001f;
			p->velocity.x = -p->velocity.x * elasticity;
		}

		if (p->position.x > -p->size + (boxWidth + center)) {
			p->position.x = -p->position.x + 2 * -(p->size - (boxWidth + center)) - 0.0001f;
			p->velocity.x = -p->velocity.x * elasticity;
		}

		if (p->position.z < p->size - boxWidth + center) {
			p->position.z = -p->position.z + 2 * (p->size - boxWidth + center) + 0.0001f;
			p->velocity.z = -p->velocity.z * elasticity;
		}

		if (p->position.z > -p->size + (boxWidth + center)) {
			p->position.z = -p->position.z + 2 * -(p->size - (boxWidth + center)) - 0.0001f;
			p->velocity.z = -p->velocity.z * elasticity;
		}
	}
}

/**
 * Compute the factor alpha_i for all particles i
 */
void parallelComputeDFSPHFactor(const SPHSystem& sphSystem, int start, int end) {
	
	for (int i = start; i < end; i++) {
		glm::vec3 alpha1 = glm::vec3(0);//第一部分
		float alpha2 = 0;//第二部分
		float alphai = 0;
		Particle* pi = sphSystem.particles[i];
		glm::ivec3 cell = sphSystem.getCell(pi);

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
					uint index = getHash(near_cell);
					Particle* pj = sphSystem.particleTable[index];
					
					// Iterate through cell linked list
					while (pj != NULL) {
						float dist2 = glm::length2(pj->position - pi->position);
						if (dist2 < sphSystem.H2 && pi != pj) {
							glm::vec3 tmp = (double)pj->mass * sphSystem.CubicKernelGradW(pi->position - pj->position);
							alpha1 += tmp;
							alpha2 += length2(tmp);
						}
						pj = pj->next;
					}
				}
			}
		}
		float mother = length2(alpha1) + alpha2;//分母
		if (mother > 0)
		{
			alphai = -pi->density / mother;
		}
		pi->alpha = alphai;
	}
}

void parallelComputeNonePressureForces(const SPHSystem& sphSystem, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* pi = sphSystem.particles[i];
		pi->force = glm::vec3(0);
		glm::ivec3 cell = sphSystem.getCell(pi);

		glm::vec3 n = glm::vec3(0);//用于计算surface force的法向
		float lapCi = 0;//Ci的拉普拉斯

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
					uint index = getHash(near_cell);
					Particle* pj = sphSystem.particleTable[index];

					// Iterate through cell linked list
					while (pj != NULL) {
						float dist2 = glm::length2(pj->position - pi->position);
						if (dist2 < sphSystem.H2 && pi != pj) {
							//unit direction and length
							float dist = sqrt(dist2);
							glm::vec3 dir = glm::normalize(pj->position - pi->position);

							//apply viscosity force
							glm::vec3 velocityDif = pj->velocity - pi->velocity;
							glm::vec3 viscoForce = (float)(pi->viscosity + pj->viscosity) / 2 * (float)pj->mass * (velocityDif / (float)pj->density) * sphSystem.SPIKY_LAP * (sphSystem.h - dist);
							pi->force += viscoForce;

							//计算ci的梯度，以确定n
							glm::vec3 nci = dir * (float)pj->mass * (pj->ci - pi->ci) / (float)pj->density * sphSystem.POLY6_GRAD;//计算norm的时候用差值
							nci *= glm::pow(sphSystem.H2 - dist2, 2) * dist;
							n += nci;

							//计算ci的拉普拉斯
							float lci = pj->mass * (pj->ci - pi->ci) / pj->density * sphSystem.POLY6_LAP * (sphSystem.H2 - dist2) * (5 * dist2 - sphSystem.H2);
							lapCi += lci;
						}
						pj = pj->next;
					}
				}
			}
		}
		//计算完之后再加上边界力
		if (dot(n, n) > 0)
		{
			glm::normalize(n);
		}
		else {
			n = glm::vec3(0);
		}
		glm::vec3 intefaceForce = -1 * sphSystem.tension * lapCi * n;
		pi->force += intefaceForce;
	}
}

void parallelPredictVelocities(const SPHSystem& sphSystem, float deltaTime, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* p = sphSystem.particles[i];

		//calculate acceleration and velocity
		glm::vec3 acceleration = p->force / p->density + glm::dvec3(0, sphSystem.g, 0);//重力怎么算？
		p->velocity += acceleration * deltaTime;
	}
}

void parallelUpdatePositions(const SPHSystem& sphSystem, float deltaTime, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* p = sphSystem.particles[i];

		// Update position
		p->position += p->velocity * (double)deltaTime;

		// Handle collisions with box
		float boxWidth = 0.8f;
		float elasticity = 0.1f;
		float center = -1.5f + (sphSystem.numParticles * sphSystem.h) / 2;
		if (p->position.y < p->size) {
			p->position.y = -p->position.y + 2 * p->size + 0.0001f;
			p->velocity.y = -p->velocity.y * elasticity;
		}

		if (p->position.x < p->size - boxWidth + center) {
			p->position.x = -p->position.x + 2 * (p->size - boxWidth + center) + 0.0001f;
			p->velocity.x = -p->velocity.x * elasticity;
		}

		if (p->position.x > -p->size + (boxWidth + center)) {
			p->position.x = -p->position.x + 2 * -(p->size - (boxWidth + center)) - 0.0001f;
			p->velocity.x = -p->velocity.x * elasticity;
		}

		if (p->position.z < p->size - boxWidth + center) {
			p->position.z = -p->position.z + 2 * (p->size - boxWidth + center) + 0.0001f;
			p->velocity.z = -p->velocity.z * elasticity;
		}

		if (p->position.z > -p->size + (boxWidth + center)) {
			p->position.z = -p->position.z + 2 * -(p->size - (boxWidth + center)) - 0.0001f;
			p->velocity.z = -p->velocity.z * elasticity;
		}
	}
}

/**
 * Parallel computation function for calculating density
 */
void parallelComputeDensity(const SPHSystem& sphSystem, int start, int end) {

	for (int i = start; i < end; i++) {
		float pDensity = 0;
		Particle* pi = sphSystem.particles[i];
		glm::ivec3 cell = sphSystem.getCell(pi);

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
					uint index = getHash(near_cell);
					Particle* pj = sphSystem.particleTable[index];

					// Iterate through cell linked list
					while (pj != NULL) {
						float dist2 = glm::length2(pj->position - pi->position);
						if (dist2 < sphSystem.H2 && pi != pj) {
							pDensity += pj->mass * sphSystem.POLY6 * glm::pow(sphSystem.H2 - dist2, 3);
						}
						pj = pj->next;
					}
				}
			}
		}

		// Include self density (as itself isn't included in neighbour)
		if (pi->type == 1)
		{
			pi->density = pDensity + sphSystem.SELF_DENS1;
		}
		else {
			pi->density = pDensity + sphSystem.SELF_DENS2;
		}
	}
}

float SPHSystem::updateTimeStepSizeCFL()
{
	//遍历找到maxvel
	float maxVel = 0.0f;
	for (int i = 0; i < particles.size(); i++) {
		Particle* pi = particles[i];
		float velMag = glm::length2(pi->velocity + (double)timeStep * pi->acceleration);
		if (velMag > maxVel)
			maxVel = velMag;
	}
	if (maxVel < 1.0e-9)
		maxVel = 1.0e-9;
	timeStep = 0.4f * (2 * h) / (sqrt(maxVel));
	timeStep = min(timeStep, DeltaTime);
	timeStep = max(timeStep, DeltaTime/10);//h = max(h, m_cflMinTimeStepSize);
	return timeStep;
}

void SPHSystem::update1(float deltaTime) {
	if (!started) return;

	deltaTime = DeltaTime;
	// Build particle hash table // init neighborhoods
	buildTable();

	// Calculate densities
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(parallelComputeDensity, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	//compute factors alphai(0)
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(parallelComputeDFSPHFactor, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	// start simulation loop
	float t = 0;
	while (t < deltaTime)
	{
		printf("t:%f\n", t);
		//compute non-pressure forces
		for (int i = 0; i < THREAD_COUNT; i++) {
			threads[i] = std::thread(parallelComputeNonePressureForces, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
		}
		for (std::thread& thread : threads) {
			thread.join();
		}

		//adapt time step size Δt according to CFL condition
		updateTimeStepSizeCFL();
		printf("timeStep:%f\n", timeStep);
		
		if (t + timeStep > deltaTime)
		{
			timeStep = deltaTime - t;
		}

		//predict velocities
		for (int i = 0; i < THREAD_COUNT; i++) {
			threads[i] = std::thread(parallelPredictVelocities, std::ref(*this), timeStep , blockBoundaries[i], blockBoundaries[i + 1]);
		}
		for (std::thread& thread : threads) {
			thread.join();
		}

		//TODO:correctDensityError 
		densitySolver();

		// update positions
		for (int i = 0; i < THREAD_COUNT; i++) {
			threads[i] = std::thread(parallelUpdatePositions, std::ref(*this), timeStep, blockBoundaries[i], blockBoundaries[i + 1]);
		}
		for (std::thread& thread : threads) {
			thread.join();
		}

		// update neighborhoods
		buildTable();

		// update ρi and αi
		// Calculate densities
		for (int i = 0; i < THREAD_COUNT; i++) {
			threads[i] = std::thread(parallelComputeDensity, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
		}
		for (std::thread& thread : threads) {
			thread.join();
		}

		//compute factors alphai
		for (int i = 0; i < THREAD_COUNT; i++) {
			threads[i] = std::thread(parallelComputeDFSPHFactor, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
		}
		for (std::thread& thread : threads) {
			thread.join();
		}

		//TODO: correctDivergenceError
		divergenceSolve();

		t += timeStep;
		//printf("t then:%f\n", t);
	}
	
}

glm::dvec3 SPHSystem::CubicKernelGradW(glm::dvec3& r)const
{
	glm::vec3 res;
	const float h3 = h * h * h;
	double m_k = 8.0f / (PI * h3);
	double m_l = 48.0f / (PI * h3);
	const double rl = sqrt(glm::length2(r));
	const float q = rl / h;
	if ((rl > 1.0e-9) && (q <= 1.0))
	{
		glm::dvec3 gradq = r / rl;
		gradq /= h;
		if (q <= 0.5)
		{
			res = m_l * q * (3.0f * q - 2.0f) * gradq;
		}
		else
		{
			const float factor = 1.0f - q;
			res = m_l * (-factor * factor) * gradq;
		}
	}
	else
		res= glm::vec3(0);

	return res;
}

void SPHSystem::divergenceSolve()
{
	int m_iter = 200;
	double m_acc = 0.001;//0.1%
	double residual = m_acc + 1; // initial residual
	double lastRes = 0;
	for (int it = 0; residual > m_acc && it < m_iter; ++it) {

		//compute density change
		double TotalDensityAdv =0;
		for (int i = 0; i < particles.size(); i++) {
				double pDensity = 0;
				Particle* pi = particles[i];
				glm::ivec3 cell = getCell(pi);

				double densityAdv = 0;
				for (int x = -1; x <= 1; x++) {
					for (int y = -1; y <= 1; y++) {
						for (int z = -1; z <= 1; z++) {
							glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
							uint index = getHash(near_cell);
							Particle* pj = particleTable[index];

							// Iterate through cell linked list
							while (pj != NULL) {
								float dist2 = glm::length2(pj->position - pi->position);
								if (dist2 < H2 && pi != pj) {
									densityAdv += -pi->density * pj->mass / pj->density * glm::dot((pi->velocity - pj->velocity),CubicKernelGradW(pi->position - pj->position));
								}
								pj = pj->next;
							}
						}
					}
				}
				pi->densityAdv = densityAdv;
				TotalDensityAdv += densityAdv;
				//这里就可以把kai算出来了
				pi->kai = 1 / timeStep * pi->densityAdv * pi->alpha;
			
		}

		// adapt velocities
		for (int i = 0; i < particles.size(); i++) {
			double pDensity = 0;
			Particle* pi = particles[i];
			glm::ivec3 cell = getCell(pi);

			glm::vec3 v = pi->velocity;
			for (int x = -1; x <= 1; x++) {
				for (int y = -1; y <= 1; y++) {
					for (int z = -1; z <= 1; z++) {
						glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
						uint index = getHash(near_cell);
						Particle* pj = particleTable[index];

						// Iterate through cell linked list
						while (pj != NULL) {
							float dist2 = glm::length2(pj->position - pi->position);
							if (dist2 < H2 ) {//&& pi != pj
								v -= timeStep * (float)pj->mass * (pj->kai / pj->density + pi->kai / pi->density) * CubicKernelGradW(pi->position - pj->position);
							}
							pj = pj->next;
						}
					}
				}
			}
			pi->velocity = v;
		}
		// Compute the new residual
		residual = TotalDensityAdv / particles.size();//avg
		printf("Pressure solver: it=%d , TotalDensityAdv=%lf\n", it, TotalDensityAdv);
		printf("Pressure solver: it=%d , res=%lf\n", it, residual);
		//printf("Pressure solver: it=%d , diff=%lf\n", it, residual- lastRes);
		//printf("Pressure solver: it=%d , bool=%d\n", it, residual==lastRes);
		if (it != 0 && (abs(lastRes - residual) < 1.0e-9 || lastRes == residual))
		{
			break;//提前退出迭代器的循环
		}
		lastRes = residual;
		residual = sqrt(residual * residual);//为了避免正负绝对值一样。最后才去和要求的误差标准比较
		
		if (it == m_iter - 1)
			printf("Pressure solver: it=%d , res=%f \n", it, residual);
		if (residual < m_acc)
			printf("Pressure solver: it=%d , res=%f, converged \n", it, residual);
	}
}

void SPHSystem::densitySolver()
{
	int m_iter = 200;
	double m_acc = 0.001;//0.1%
	double residual = m_acc + 1; // initial residual

	double lastRes = 0;
	for (int it = 0; residual > m_acc && it < m_iter; ++it) {
		//compute density change
		double TotalDensityError = 0;
		for (int i = 0; i < particles.size(); i++) {
			double pDensity = 0;
			Particle* pi = particles[i];
			glm::ivec3 cell = getCell(pi);

			double densityAdv = 0;
			for (int x = -1; x <= 1; x++) {
				for (int y = -1; y <= 1; y++) {
					for (int z = -1; z <= 1; z++) {
						glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
						uint index = getHash(near_cell);
						Particle* pj = particleTable[index];

						// Iterate through cell linked list
						while (pj != NULL) {
							float dist2 = glm::length2(pj->position - pi->position);
							if (dist2 < H2 && pi != pj) {
								densityAdv += -pi->density * pj->mass / pj->density * glm::dot((pi->velocity - pj->velocity), CubicKernelGradW(pi->position - pj->position));
							}
							pj = pj->next;
						}
					}
				}
			}
			pi->rho_predict = pi->density + timeStep * densityAdv;
			float Rdens = pi->type == 1 ? restDensity1 : restDensity2;
			TotalDensityError += pi->rho_predict - Rdens;
			pi->kai = (pi->rho_predict - Rdens) / (timeStep * timeStep) * pi->alpha;

		}

		// adapt velocities
		for (int i = 0; i < particles.size(); i++) {
			double pDensity = 0;
			Particle* pi = particles[i];
			glm::ivec3 cell = getCell(pi);

			glm::vec3 v = pi->velocity;
			for (int x = -1; x <= 1; x++) {
				for (int y = -1; y <= 1; y++) {
					for (int z = -1; z <= 1; z++) {
						glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
						uint index = getHash(near_cell);
						Particle* pj = particleTable[index];

						// Iterate through cell linked list
						while (pj != NULL) {
							float dist2 = glm::length2(pj->position - pi->position);
							if (dist2 < H2) {// && pi != pj
								v -= timeStep * pj->mass * (pj->kai / pj->density + pi->kai / pi->density) * CubicKernelGradW(pi->position - pj->position);
							}
							pj = pj->next;
						}
					}
				}
			}
			pi->velocity = v;
		}

		// Compute the new residual
		residual = TotalDensityError / particles.size();//avg
		printf("density solver : it=%d , TotalDensityError=%f\n", it, TotalDensityError);
		
		printf("density solver : it=%d , res=%f\n", it, residual);
		if (it != 0 && lastRes == residual)
		{
			break;//提前退出迭代器的循环
		}
		lastRes = residual;
		residual = sqrt(residual * residual);
		if (it == m_iter - 1)
			printf("density solver: it=%d , res=%f \n", it, residual);
		if (residual < m_acc)
			printf("density solver: it=%d , res=%f, converged \n", it, residual);


	}
}


void SPHSystem::update(float deltaTime) {
	if (!started) return;

	// To increase system stability, a fixed deltaTime is set
	deltaTime = DeltaTime;

	// Build particle hash table
	buildTable();

	// Calculate densities and pressures of particles
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(parallelDensityAndPressures, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	// Calclulate forces of particles
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(parallelForces, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	// Update positions of all particles
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(parallelUpdateParticlePositions, std::ref(*this), deltaTime, blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}
}

void SPHSystem::draw(const glm::mat4& viewProjMtx, GLuint shader) {

	//生成颜色数组
	std::vector<glm::vec3> Colors;
	for (int i = 0; i < particles.size(); i++) {
		if (particles[i]->type ==1)
		{
			Colors.push_back(glm::vec3(0.9f, 0.1f, 0.0f));
		}
		else {
			Colors.push_back(glm::vec3(0.0f, 0.5f, 0.9f));
		}
	}
	// Generate VBO for sphere model matrices
	GLuint cvbo;//, vao,ebo
	glGenBuffers(1, &cvbo);
	glBindBuffer(GL_ARRAY_BUFFER, cvbo);
	glBufferData(GL_ARRAY_BUFFER, Colors.size() * sizeof(glm::vec3), Colors.data(), GL_STATIC_DRAW);

	// Setup instance VAO
	glBindVertexArray(sphere->vao);
	glEnableVertexAttribArray(6);
	glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), 0);
	glVertexAttribDivisor(6, 1);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	// Calculate model matrices for each particle
	for (int i = 0; i < particles.size(); i++) {
		glm::mat4 translate = glm::translate(particles[i]->position);
		sphereModelMtxs[i] = translate * sphereScale;
	}

	// Send matrix data to GPU
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	void* data = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	memcpy(data, sphereModelMtxs, sizeof(glm::mat4) * particles.size());
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Draw instanced particles
	//int vertexColorLocation = glGetUniformLocation(shader, "DiffuseColor");
	glUseProgram(shader);
	//glUniform3f(vertexColorLocation, 0.9f, 0.1f, 0.0f);
	glUniformMatrix4fv(glGetUniformLocation(shader, "viewProjMtx"), 1, false, (float*)&viewProjMtx);
	glBindVertexArray(sphere->vao);
	glDrawElementsInstanced(GL_TRIANGLES, sphere->indices.size(), GL_UNSIGNED_INT, 0, particles.size());
	glBindVertexArray(0);
	glUseProgram(0);
}

/**
 * Parallel helper for clearing table 
 */
void tableClearHelper(SPHSystem& sphSystem, int start, int end) {
	for (int i = start; i < end; i++) {
		sphSystem.particleTable[i] = NULL;
	}
}

/**
 * Parallel helper for building table 
 */
void buildTableHelper(SPHSystem& sphSystem, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* pi = sphSystem.particles[i];

		// Calculate hash index using hashing formula
		uint index = getHash(sphSystem.getCell(pi));

		// Setup linked list if need be
		mtx.lock();
		if (sphSystem.particleTable[index] == NULL) {
			pi->next = NULL;
			sphSystem.particleTable[index] = pi;
		}
		else {
			pi->next = sphSystem.particleTable[index];
			sphSystem.particleTable[index] = pi;
		}
		mtx.unlock();
	}
}

void SPHSystem::buildTable() {
	// Parallel empty the table
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(tableClearHelper, std::ref(*this), tableBlockBoundaries[i], tableBlockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	// Parallel build table
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(buildTableHelper, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}
}

glm::ivec3 SPHSystem::getCell(Particle* p) const {
	return glm::ivec3(p->position.x / h, p->position.y / h, p->position.z / h);
}

void SPHSystem::reset() {
	initParticles();
	started = false;
}

void SPHSystem::startSimulation() {
	started = true;
}

void SPHSystem::pause() {
	started = false;
}

void SPHSystem::single() {
	started = true;
	SPHSystem::update(0.003);
	started = false;
}