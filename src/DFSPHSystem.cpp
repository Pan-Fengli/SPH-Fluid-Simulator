#include "DFSPHSystem.h"

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

// This will lock across all instances of DFSPHSystem's,
// however since we only really have one instance, this
// should be okay for now  
std::mutex dfmtx;

/**
 * Hashes the position of a cell, giving where
 * the cell goes in the particle table
 */
uint dfgetHash(const glm::ivec3& cell) {
	return (
		(uint)(cell.x * 73856093)
		^ (uint)(cell.y * 19349663)
		^ (uint)(cell.z * 83492791)
		) % TABLE_SIZE;
}

DFSPHSystem::DFSPHSystem(unsigned int numParticles, float mass1, float mass2, double restDensity1, double restDensity2, float gasConst, double viscosity1, double viscosity2, double h, double g, double tension) {
	this->numParticles = numParticles;
	this->restDensity1 = restDensity1;
	this->restDensity2 = restDensity2;
	this->viscosity1 = viscosity1;
	this->viscosity2 = viscosity2;
	this->dx = h;
	double df_fac = 1.3;
	this->h = h * df_fac;
	this->g = g;
	this->tension = tension;

	POLY6 = 315.0f / (64.0f * PI * pow(h, 9));
	SPIKY_GRAD = -45.0f / (PI * pow(h, 6));
	SPIKY_LAP = 45.0f / (PI * pow(h, 6));
	POLY6_GRAD = -945.0f / (32.0f * PI * pow(h, 9));
	POLY6_LAP = 945.0f / (32.0f * PI * pow(h, 9));
	//MASS1 = mass1;
	//MASS2 = mass2;
	int factor = 0.70789;
	MASS1 = pow(dx, 3) * restDensity1;
	MASS2 = pow(dx, 3) * restDensity2;
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
	sphereScale = glm::scale(glm::vec3(h / 2.f));
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
	glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(sizeof(glm::vec4) * 2));
	glEnableVertexAttribArray(5);
	glVertexAttribPointer(5, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(glm::vec4), (void*)(sizeof(glm::vec4) * 3));

	glVertexAttribDivisor(2, 1);
	glVertexAttribDivisor(3, 1);
	glVertexAttribDivisor(4, 1);
	glVertexAttribDivisor(5, 1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	//start init
	started = false;

	// Allocate table memory
	particleTable = (Particle**)malloc(sizeof(Particle*) * TABLE_SIZE);

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

DFSPHSystem::~DFSPHSystem() {
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

void DFSPHSystem::initParticles() {
	std::srand(1024);
	//double h = dx;
	float particleSeperation = h + 0.01f;
	particleSeperation *= 1.276;
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
				Particle* nParticle;
				//初始化两种粒子
				if (pcount < size / 2)
				{
					//红色的粒子
					nParticlePos.y -= 0.05f;//0.05
					nParticlePos.x -= particleSeperation;
					nParticle = new Particle(MASS1, h, nParticlePos, glm::dvec3(0), 1, 0.5);
					nParticle->viscosity = viscosity1;
				}
				else {
					//蓝色的粒子
					nParticlePos.y -= 0.05f;
					nParticlePos.x += particleSeperation;
					nParticle = new Particle(MASS2, h, nParticlePos, glm::dvec3(0), 2, -0.5);
					nParticle->viscosity = viscosity2;
				}

				//append particle
				particles[i + (j + numParticles * k) * numParticles] = nParticle;
				pcount++;
			}
		}
	}
}

void parallelComputeNonePressureForces(const DFSPHSystem& sphSystem, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* pi = sphSystem.particles[i];
		glm::ivec3 cell = sphSystem.getCell(pi);

		glm::dvec3 d_v = glm::dvec3(0);//dv/dt
		glm::dvec3 pos_i = pi->position;

		glm::dvec3 n = glm::dvec3(0);//用于计算surface force的法向
		float lapCi = 0;//Ci的拉普拉斯

		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
					uint index = dfgetHash(near_cell);
					Particle* pj = sphSystem.particleTable[index];

					// Iterate through cell linked list
					while (pj != NULL) {
						float dist2 = glm::length2(pj->position - pi->position);
						glm::dvec3 pos_j = pj->position;
						glm::dvec3 r = pos_i - pos_j;
						double r_mod = length(r);
						if (r_mod > 1e-4 && pi != pj) {
							//unit direction and length
							float dist = sqrt(dist2);
							glm::dvec3 dir = glm::normalize(pj->position - pi->position);

							//apply viscosity force
							double v_xy = dot(pi->velocity - pj->velocity, r);
							if (v_xy < 0)
							{
								double vmu = 2.0 * (pi->viscosity + pj->viscosity) / 2 * sphSystem.dx * sphSystem.c_0 / (pi->density + pj->density);
								double sab = -vmu * v_xy / (pow(r_mod, 2) + 0.01 * pow(sphSystem.dx, 2));
								glm::dvec3 res = -pj->mass * sab * sphSystem.cubic_kernel_derivative(r_mod, sphSystem.h) * r / r_mod;
								d_v += res;
							}

							//计算界面力——
							if (dist2 < sphSystem.H2 && pi != pj) {
								//计算ci的梯度，以确定n
								glm::dvec3 nci = dir * pj->mass * ((double)pj->ci - pi->ci) / pj->density * sphSystem.POLY6_GRAD;//计算norm的时候用差值
								nci *= glm::pow(sphSystem.H2 - dist2, 2) * dist;
								n += nci;

								//计算ci的拉普拉斯
								float lci = pj->mass * (pj->ci - pi->ci) / pj->density * sphSystem.POLY6_LAP * (sphSystem.H2 - dist2) * (5 * dist2 - sphSystem.H2);
								lapCi += lci;
							}
						}
						pj = pj->next;
					}
				}
			}
		}
		//printf("visco force:%f,%f,%f\n", d_v.x, d_v.y, d_v.z);
		//计算完之后再加上边界力
		if (dot(n, n) > 0)
		{
			glm::normalize(n);
		}
		else {
			n = glm::vec3(0);
		}
		glm::dvec3 intefaceForce = -1 * sphSystem.tension * lapCi * n;
		d_v += intefaceForce / pi->density;
		//printf("intefaceForce force:%f,%f,%f\n", d_v.x, d_v.y, d_v.z);
		//printf("pi->acceleration except gravity: %f\n", d_v.y);
		//Add body force 重力
		d_v += glm::dvec3(0, sphSystem.g, 0);

		pi->acceleration = d_v;
		//printf("pi->acceleration: %f,%f,%f\n", pi->acceleration.x, pi->acceleration.y, pi->acceleration.z);
	}
}

void parallelPredictVelocities(const DFSPHSystem& sphSystem, float deltaTime, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* p = sphSystem.particles[i];

		//calculate acceleration and velocity
		p->particle_velocity_new += p->acceleration * sphSystem.timeStep;
		//printf("pi->acceleration: %f,%f,%f\n", p->acceleration.x, p->acceleration.y, p->acceleration.z);
		//printf(" PredictVelocities pi->particle_velocity_new: %f,%f,%f\n", p->particle_velocity_new.x, p->particle_velocity_new.y, p->particle_velocity_new.z);
	}
}

void handleCollision(const DFSPHSystem& sphSystem, float deltaTime, int start, int end)
{
	for (int i = start; i < end; i++) {
		Particle* p = sphSystem.particles[i];
		// Handle collisions with box
		float boxWidth = 3.2f;
		float elasticity = 0.001f;
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

void parallelUpdatePositions(const DFSPHSystem& sphSystem, float deltaTime, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* p = sphSystem.particles[i];
		// Update position
		p->position += p->particle_velocity_new * sphSystem.timeStep;
		//printf("pi->position: %f,%f,%f\n", p->position.x, p->position.y, p->position.z);
		//printf("pi->particle_velocity_new: %f,%f,%f\n", p->particle_velocity_new.x, p->particle_velocity_new.y, p->particle_velocity_new.z);

	}
}

void parallelUpdateVelocity(const DFSPHSystem& sphSystem, float deltaTime, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* p = sphSystem.particles[i];
		// Update position
		p->velocity = p->particle_velocity_new;
	}
}


/**
 * Parallel computation function for calculating density
 */
void parallelComputeDensityAndAplha(const DFSPHSystem& sphSystem, int start, int end) {

	for (int i = start; i < end; i++) {
		Particle* pi = sphSystem.particles[i];
		glm::ivec3 cell = sphSystem.getCell(pi);

		glm::dvec3 grad_sum = glm::dvec3(0);
		double grad_square_sum = 0;
		double curr_rho = 0;
		glm::dvec3 pos_i = pi->position;
		for (int x = -1; x <= 1; x++) {
			for (int y = -1; y <= 1; y++) {
				for (int z = -1; z <= 1; z++) {
					glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
					uint index = dfgetHash(near_cell);
					Particle* pj = sphSystem.particleTable[index];

					// Iterate through cell linked list
					while (pj != NULL) {
						glm::dvec3 pos_j = pj->position;
						glm::dvec3 r = pos_i - pos_j;
						double r_mod = length(r);
						if (r_mod > 1e-4) {
							glm::dvec3 grad_val = pj->mass * sphSystem.cubic_kernel_derivative(r_mod, sphSystem.h) * r / r_mod;
							grad_sum += grad_val;
							grad_square_sum += dot(grad_val, grad_val);

							// Compute the density
							curr_rho += pj->mass * sphSystem.cubic_kernel(r_mod, sphSystem.h);
							//换回原来的计算方式——
							//float dist2 = glm::length2(pj->position - pi->position);
							//if (dist2 < sphSystem.H2 && pi != pj) {
							//	curr_rho += pj->mass * sphSystem.POLY6 * glm::pow(sphSystem.H2 - dist2, 3);
							//}
						}
						pj = pj->next;
					}
				}
			}
		}

		// Include self density (as itself isn't included in neighbour)
		//if (pi->type == 1)
		//{
		//	pi->density = curr_rho + sphSystem.SELF_DENS1;
		//}
		//else {
		//	pi->density = curr_rho + sphSystem.SELF_DENS2;
		//}
		pi->density = curr_rho;
		//Set a threshold of 10 ^ -6 to avoid instability
		pi->alpha = -1.0 / max(dot(grad_sum, grad_sum) + grad_square_sum, 1e-6);
	}
}

float DFSPHSystem::updateTimeStepSizeCFL()
{
	//遍历找到maxvel
	double maxVel = 0.0f;
	double max_a = 0;
	for (int i = 0; i < particles.size(); i++) {
		Particle* pi = particles[i];
		double velMag = length(pi->velocity);
		double aMag = length(pi->acceleration + pi->particle_pressure_acc);
		if (velMag > maxVel)
			maxVel = velMag;
		if (aMag > max_a)
			max_a = aMag;
	}
	// CFL analysis, constrained by v_max
	double dt_v = CFL_v * h / max(maxVel, 1e-5);
	// Constrained by a_max
	double dt_a = CFL_a * sqrt(h / max(max_a, 1e-5));

	timeStep = min(dt_v, min(dt_a, 0.0005));
	printf("timeStep:%f\n", timeStep);
	return timeStep;
}

void DFSPHSystem::update1(float deltaTime) {//step
	if (!started) return;

	deltaTime = DeltaTime;
	// Build particle hash table // init neighborhoods
	buildTable();

	// Calculate densities
	//compute_density_alpha, update ρi and αi
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(parallelComputeDensityAndAplha, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	// start simulation loop
	float t = 0;
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

	//predict velocities
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(parallelPredictVelocities, std::ref(*this), timeStep, blockBoundaries[i], blockBoundaries[i + 1]);
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
		threads[i] = std::thread(parallelComputeDensityAndAplha, std::ref(*this), blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	//TODO: correctDivergenceError
	divergenceSolve();

	//Update velocities v
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(parallelUpdateVelocity, std::ref(*this), timeStep, blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

	//hanlde collision
	for (int i = 0; i < THREAD_COUNT; i++) {
		threads[i] = std::thread(handleCollision, std::ref(*this), timeStep, blockBoundaries[i], blockBoundaries[i + 1]);
	}
	for (std::thread& thread : threads) {
		thread.join();
	}

}

double DFSPHSystem::cubic_kernel_derivative(double r_mod, double h)const
{
	double k = 1. / PI / pow(h, 3);
	double q = r_mod / h;
	double res = 0;
	if (0 < q < 1.0)
		res = (k / h) * (-3 * q + 2.25 * pow(q, 2));
	else if (q < 2.0)
		res = -0.75 * (k / h) * pow((2 - q), 2);

	return res;
}

double DFSPHSystem::cubic_kernel(double r_mod, double h)const
{
	double k = 1. / PI / pow(h, 3);
	double q = r_mod / h;
	double res = 0;
	if (0 < q <= 1.0)
		res = k * (1 - 1.5 * pow(q, 2) + 0.75 * pow(q, 3));
	else if (q < 2.0)
		res = k * 0.25 * pow((2 - q), 3);
	return res;
}

void DFSPHSystem::divergenceSolve()
{
	int m_iter = 1000;
	double m_acc = 0.01;//1%
	double residual = m_acc + 1; // initial residual
	double lastRes = 0;
	int it_div = 0;
	double sum_drho = 0;
	while (sum_drho >= m_acc * particles.size() * (restDensity1 + restDensity2) / 2
		|| it_div < 1)
	{
		//compute stiffness parameter,correct_divergence_compute_drho
		for (int i = 0; i < particles.size(); i++) {
			Particle* pi = particles[i];
			glm::ivec3 cell = getCell(pi);

			glm::dvec3 pos_i = pi->position;
			double d_rho = 0;
			for (int x = -1; x <= 1; x++) {
				for (int y = -1; y <= 1; y++) {
					for (int z = -1; z <= 1; z++) {
						glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
						uint index = dfgetHash(near_cell);
						Particle* pj = particleTable[index];
						// Iterate through cell linked list
						while (pj != NULL) {
							glm::dvec3 pos_j = pj->position;
							glm::dvec3 r = pos_i - pos_j;
							double r_mod = length(r);
							if (r_mod > 1e-4 && pi != pj) {
								d_rho += pj->mass * cubic_kernel_derivative(r_mod, h)
									* dot((pi->particle_velocity_new - pj->particle_velocity_new), (r / r_mod));
							}
							pj = pj->next;
						}
					}
				}
			}
			pi->d_density = max(d_rho, 0.0);
			//pi->d_density = d_rho;
			double Rdens = pi->type == 1 ? restDensity1 : restDensity2;
			if (pi->density + timeStep * pi->d_density < Rdens
				&& pi->density < Rdens)
			{
				pi->d_density = 0.0;
			}
			pi->particle_stiff = pi->d_density * pi->alpha;
			sum_drho += pi->d_density;	
		}

		//correct_divergence_adapt_vel
		for (int i = 0; i < particles.size(); i++) {
			Particle* pi = particles[i];
			glm::ivec3 cell = getCell(pi);

			glm::dvec3 pos_i = pi->position;
			glm::dvec3 d_v= glm::dvec3(0);
			for (int x = -1; x <= 1; x++) {
				for (int y = -1; y <= 1; y++) {
					for (int z = -1; z <= 1; z++) {
						glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
						uint index = dfgetHash(near_cell);
						Particle* pj = particleTable[index];
						// Iterate through cell linked list
						while (pj != NULL) {
							glm::dvec3 pos_j = pj->position;
							glm::dvec3 r = pos_i - pos_j;
							double r_mod = length(r);
							if (r_mod > 1e-5 && pi != pj) {
								d_v += pj->mass * (pi->particle_stiff + pj->particle_stiff)
									* cubic_kernel_derivative(r_mod, h) * r / r_mod;
							}
							pj = pj->next;
						}
					}
				}
			}
			pi->particle_velocity_new += d_v;
			pi->particle_pressure_acc = d_v / timeStep;
			double Rdens = pi->type == 1 ? restDensity1 : restDensity2;
			pi->pressure = pi->particle_stiff * -pi->density / timeStep * (pi->density - Rdens);
		}

		it_div++;
		if (it_div > m_iter)
		{
			printf("Warning: DFSPH divergence does not converge, iterated %d steps\n", it_div);
			break;
		}
	}
	printf("divergence solver : it=%d , res=%f\n", it_div, sum_drho);

}

void DFSPHSystem::densitySolver()
{
	int m_iter = 1000;
	double m_acc = 0.01;//1%
	int it_density = 0;
	double sum_rho_err = 0;
	while (sum_rho_err >= m_acc * particles.size() * (restDensity1 + restDensity2) / 2
		|| it_density <2)
	{
		sum_rho_err = 0;

		//correct_density_predict
		for (int i = 0; i < particles.size(); i++) {
			Particle* pi = particles[i];
			glm::ivec3 cell = getCell(pi);

			glm::dvec3 pos_i = pi->position;
			double d_rho = 0;
			for (int x = -1; x <= 1; x++) {
				for (int y = -1; y <= 1; y++) {
					for (int z = -1; z <= 1; z++) {
						glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
						uint index = dfgetHash(near_cell);
						Particle* pj = particleTable[index];
						// Iterate through cell linked list
						while (pj != NULL) {
							glm::dvec3 pos_j = pj->position;
							glm::dvec3 r = pos_i - pos_j;
							double r_mod = max(length(r), 1e-5);
							if (pi != pj) {
								d_rho += pj->mass * cubic_kernel_derivative(r_mod, h)
									* dot((pi->particle_velocity_new - pj->particle_velocity_new), (r / r_mod));
							}
							pj = pj->next;
						}
					}
				}
			}

			//Compute the predicted density rho star
			pi->rho_predict = pi->density + d_rho * timeStep;
			double Rdens = pi->type == 1 ? restDensity1 : restDensity2;
			double err = max(0.0, pi->rho_predict - Rdens);
			pi->particle_stiff = err * pi->alpha;
			sum_rho_err += err;
		}

		//correct_density_adapt_vel # predict velocity for correct density error
		for (int i = 0; i < particles.size(); i++) {
			Particle* pi = particles[i];
			glm::ivec3 cell = getCell(pi);

			glm::dvec3 pos_i = pi->position;
			glm::dvec3 d_v = glm::dvec3(0);
			for (int x = -1; x <= 1; x++) {
				for (int y = -1; y <= 1; y++) {
					for (int z = -1; z <= 1; z++) {
						glm::ivec3 near_cell = cell + glm::ivec3(x, y, z);
						uint index = dfgetHash(near_cell);
						Particle* pj = particleTable[index];
						// Iterate through cell linked list
						while (pj != NULL) {
							glm::dvec3 pos_j = pj->position;
							glm::dvec3 r = pos_i - pos_j;
							double r_mod = length(r);
							if(i==10) printf("%f :r: %f,%f,%f\n", r_mod, r.x, r.y, r.z);
							if (r_mod > 1e-4 && pi != pj) {
								d_v += pj->mass * (pi->particle_stiff + pj->particle_stiff) * cubic_kernel_derivative(r_mod, h) * r / r_mod;
							}
							pj = pj->next;
						}
					}
				}
			}
			pi->particle_velocity_new += d_v / max(timeStep, 1e-5);
			//printf("&d :d_v: %f,%f,%f\n",i, d_v.x, d_v.y, d_v.z);
			pi->particle_pressure_acc = d_v / max(timeStep * timeStep, 1e-8);
		}
		it_density++;
		if (it_density > m_iter)
		{
			printf("Warning: DFSPH density does not converge, iterated %d steps", it_density);
			break;
		}
	}
	printf("density solver : it=%d , res=%f\n", it_density, sum_rho_err);
}

void DFSPHSystem::draw(const glm::mat4& viewProjMtx, GLuint shader) {

	//生成颜色数组
	std::vector<glm::vec3> Colors;
	for (int i = 0; i < particles.size(); i++) {
		if (particles[i]->type == 1)
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
void tableClearHelper(DFSPHSystem& sphSystem, int start, int end) {
	for (int i = start; i < end; i++) {
		sphSystem.particleTable[i] = NULL;
	}
}

/**
 * Parallel helper for building table
 */
void buildTableHelper(DFSPHSystem& sphSystem, int start, int end) {
	for (int i = start; i < end; i++) {
		Particle* pi = sphSystem.particles[i];

		// Calculate hash index using hashing formula
		uint index = dfgetHash(sphSystem.getCell(pi));

		// Setup linked list if need be
		dfmtx.lock();
		if (sphSystem.particleTable[index] == NULL) {
			pi->next = NULL;
			sphSystem.particleTable[index] = pi;
		}
		else {
			pi->next = sphSystem.particleTable[index];
			sphSystem.particleTable[index] = pi;
		}
		dfmtx.unlock();
	}
}

void DFSPHSystem::buildTable() {
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

glm::ivec3 DFSPHSystem::getCell(Particle* p) const {
	return glm::ivec3(p->position.x / h, p->position.y / h, p->position.z / h);
}

void DFSPHSystem::reset() {
	initParticles();
	started = false;
}

void DFSPHSystem::startSimulation() {
	started = true;
}

void DFSPHSystem::pause() {
	started = false;
}

void DFSPHSystem::single() {
	started = true;
	DFSPHSystem::update1(0.003);
	started = false;
}