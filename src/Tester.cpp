////////////////////////////////////////
// Tester.cpp
////////////////////////////////////////

#include "Tester.h"
#include <iostream>

#include "imgui/imgui.h"
#include "imgui/imgui_impl_glut.h"
#include "imgui/imgui_impl_opengl2.h"

////////////////////////////////////////////////////////////////////////////////

static Tester *TESTER=0;

int main(int argc, char **argv) {
	glutInit(&argc, argv);

	TESTER=new Tester("SPH Fluid Simulator",argc,argv);
	glutMainLoop();
	delete TESTER;

	return 0;
}

////////////////////////////////////////////////////////////////////////////////

// These are really HACKS to make glut call member functions instead of static functions
static void display()									{TESTER->Draw();}
static void idle()										{TESTER->Update();}
static void resize(int x,int y)							{TESTER->Resize(x,y);}
static void keyboard(unsigned char key,int x,int y)		{TESTER->Keyboard(key,x,y);}
static void specialKeys(int key, int x, int y) { TESTER->SpecialKeys(key, x, y); }
static void mousebutton(int btn,int state,int x,int y)	{TESTER->MouseButton(btn,state,x,y);}
static void mousemotion(int x, int y)					{TESTER->MouseMotion(x,y);}
static void mouseWheel(int button, int dir, int x, int y){ TESTER->MouseWheel(button, dir, x, y);};

////////////////////////////////////////////////////////////////////////////////

Tester::Tester(const char *windowTitle,int argc,char **argv) {
	WinX=1280;
	WinY=720;
	LeftDown=MiddleDown=RightDown=false;
	MouseX=MouseY=0;
	prevTime = 0;
	currentTime = 0;
	deltaTime = 0;

	// Create the window
	glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );
	glutInitWindowSize( WinX, WinY );
	glutInitWindowPosition( 100, 100 );
	WindowHandle = glutCreateWindow( windowTitle );
	glutSetWindowTitle( windowTitle );
	glutSetWindow( WindowHandle );

	// Background color
	glClearColor( 0.75, 0.75, 0.75, 1. );

	// Callbacks
	glutDisplayFunc( display );
	glutIdleFunc( idle );
	glutKeyboardFunc( keyboard );
	glutMouseFunc( mousebutton );
	glutMotionFunc( mousemotion );
	glutMouseWheelFunc(mouseWheel);
	glutPassiveMotionFunc( mousemotion );
	glutReshapeFunc( resize );

	// Initialize GLEW
	glewInit();
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	// Setup Dear ImGui context
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	// Setup Dear ImGui style
	ImGui::StyleColorsDark();

	// Setup Platform/Renderer bindings
	ImGui_ImplGLUT_Init();
	ImGui_ImplGLUT_InstallFuncs();
	ImGui_ImplOpenGL2_Init();

	// Initialize components
	InstanceProgram=new ShaderProgram("resources/Instance.glsl",ShaderProgram::eRender);
	Cam=new Camera;
	Cam->SetAspect(float(WinX)/float(WinY));

	//init SPH system
	sphSystem = new SPHSystem(15, 0.025f, 0.015f, 1000, 800, 1, 0.84f, 0.54f, 0.15f, -9.8f, 0.2f);
	//sphSystem = new DFSPHSystem(15, 0.025f, 0.015f, 1000, 800, 1, 0.84f, 0.54f, 0.15f, -9.8f, 0.2f);

}

////////////////////////////////////////////////////////////////////////////////

Tester::~Tester() {
	delete Cam;

	// Cleanup imgui
	ImGui_ImplOpenGL2_Shutdown();
	ImGui_ImplGLUT_Shutdown();
	ImGui::DestroyContext();

	glFinish();
	glutDestroyWindow(WindowHandle);
}

////////////////////////////////////////////////////////////////////////////////

void Tester::Update() {
	//calculate delta time
	currentTime = glutGet(GLUT_ELAPSED_TIME);
	deltaTime = (currentTime - prevTime)/1000.f;
	prevTime = currentTime;

	// Update the components in the world
	Cam->Update();
	
	//update sph system
	sphSystem->update(deltaTime);
	
	// Tell glut to re-display the scene
	glutSetWindow(WindowHandle);
	glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////

void Tester::Reset() {
	Cam->Reset();
	Cam->SetAspect(float(WinX)/float(WinY));
}

////////////////////////////////////////////////////////////////////////////////

void Tester::Draw() {
	// Begin drawing scene
	glViewport(0, 0, WinX, WinY);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Draw sph system
	sphSystem->draw(Cam->GetViewProjectMtx(), InstanceProgram->GetProgramID());

	// Render GUI
	ImGui_ImplOpenGL2_NewFrame();
	ImGui_ImplGLUT_NewFrame();
	{
		static int numParticles = 22;//22
		static float nMass1 = 0.1;//0.015
		static float nMass2 = 0.01;//0.003
		static float nh = 0.045;//0.1,0.045 
		static float nRest1 = 1000.f;//1000
		static float nRest2 = 100.f;//200
		static float nVisco1 = 15.f;//5,50
		static float nVisco2 = 15.f;
		static float gasConst = 20.f;
		static float tension = 1.0f;
		static int counter = 0;

		ImGui::Begin("SPH debug");                          // Create GUI window

		ImGui::Text("Change values for the simulation. Press RESET to commit changes"); 

		ImGui::SliderInt("Number of Particles", &numParticles, 10, 100);            // Edit number of particles
		ImGui::SliderFloat("Mass of Particle 1", &nMass1, 0.001f, 0.5f);            // Edit mass
		ImGui::SliderFloat("Mass of Particle 2", &nMass2, 0.001f, 0.5f);            // Edit mass
		ImGui::SliderFloat("Support Radius", &nh, 0.001f, 1.f);            // Edit support radius
		ImGui::SliderFloat("Rest Density 1", &nRest1, 0.001f, 2000.f);            // Edit rest density
		ImGui::SliderFloat("Rest Density 2", &nRest2, 0.001f, 2000.f);            // Edit rest density
		ImGui::SliderFloat("Viscosity Constant 1", &nVisco1, 0.001f, 20.f);            // Edit viscosity
		ImGui::SliderFloat("Viscosity Constant 2", &nVisco2, 0.001f, 20.f);            // Edit viscosity
		ImGui::SliderFloat("Gas Constant", &gasConst, 0.001f, 3.f);            // Edit gas constant
		ImGui::SliderFloat("Tension", &tension, 0.001f, 3.f);            // ��������

		if (ImGui::Button("RESET")) {
			delete sphSystem;
			sphSystem = new SPHSystem(numParticles, nMass1, nMass2, nRest1,nRest2, gasConst, nVisco1, nVisco2, nh, -9.8, tension);
		}

		if (ImGui::Button("START")) {
			if (sphSystem != NULL)
			{
				sphSystem->startSimulation();
			}
		}
		if (ImGui::Button("PAUSE")) {
			if (sphSystem != NULL)
			{
				sphSystem->pause();
			}
		}
		if (ImGui::Button("SINGLE")) {
			if (sphSystem != NULL)
			{
				sphSystem->single();
			/*	sphSystem->startSimulation();
				sphSystem->update(deltaTime);
				sphSystem->pause();*/
			}
		}

		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::End();
	}
	glUseProgram(0);
	ImGui::Render();
	ImGui_ImplOpenGL2_RenderDrawData(ImGui::GetDrawData());

	// Finish drawing scene
	glFinish();
	glutSwapBuffers();
}

////////////////////////////////////////////////////////////////////////////////

void Tester::Quit() {
	glFinish();
	glutDestroyWindow(WindowHandle);
	exit(0);
}

////////////////////////////////////////////////////////////////////////////////

void Tester::Resize(int x,int y) {
	WinX = x;
	WinY = y;
	Cam->SetAspect(float(WinX)/float(WinY));

	ImGui_ImplGLUT_ReshapeFunc(x, y);
}

////////////////////////////////////////////////////////////////////////////////

void Tester::Keyboard(int key,int x,int y) {
	ImGui_ImplGLUT_KeyboardFunc(key, x, y);

	switch(key) {
		case 0x1b:		// Escape
			Quit();
			break;
		case 'r':
			Reset();
			sphSystem->reset();
			break;
		case 'c':
			sphSystem->startSimulation();
			break;
	}
}

////////////////////////////////////////////////////////////////////////////////

void Tester::SpecialKeys(int key, int x, int y) {
	switch (key) {	
		
	}
}
////////////////////////////////////////////////////////////////////////////////

void Tester::MouseWheel(int button, int dir, int x, int y)
{
	if (dir > 0)
	{
		const float rate = 0.0005f;
		float dist = glm::clamp(Cam->GetDistance() * (1.0f - x * rate), 0.01f, 1000.0f);
		Cam->SetDistance(dist);
	}
	else
	{
		// Zoom out
		const float rate = 0.0005f;
		float dist = glm::clamp(Cam->GetDistance() * (1.0f + x * rate), 0.01f, 1000.0f);
		Cam->SetDistance(dist);
	}

	return;
}

////////////////////////////////////////////////////////////////////////////////

void Tester::MouseButton(int btn,int state,int x,int y) {
	// Send mouse inputs to GUI
	ImGui_ImplGLUT_MouseFunc(btn, state, x, y);

	// Don't read input if mouse inside GUI
	if (ImGui::GetIO().WantCaptureMouse)
		return;

	if(btn==GLUT_LEFT_BUTTON) {
		LeftDown = (state==GLUT_DOWN);
	}
	else if(btn==GLUT_MIDDLE_BUTTON) {
		MiddleDown = (state==GLUT_DOWN);
	}
	else if(btn==GLUT_RIGHT_BUTTON) {
		RightDown = (state==GLUT_DOWN);
	}
}

////////////////////////////////////////////////////////////////////////////////

void Tester::MouseMotion(int nx,int ny) {
	// Send mouse inputs to GUI
	ImGui_ImplGLUT_MotionFunc(nx, ny);

	// Don't read input if mouse inside GUI
	if (ImGui::GetIO().WantCaptureMouse)
		return;

	int maxDelta=100;
	int dx = glm::clamp(nx - MouseX,-maxDelta,maxDelta);
	int dy = glm::clamp(-(ny - MouseY),-maxDelta,maxDelta);

	MouseX = nx;
	MouseY = ny;

	// Move camera
	// NOTE: this should really be part of Camera::Update()
	if(LeftDown) {
		const float rate=1.0f;
		Cam->SetAzimuth(Cam->GetAzimuth()+dx*rate);
		Cam->SetIncline(glm::clamp(Cam->GetIncline()-dy*rate,-90.0f,90.0f));
	}
	if(RightDown) {
		const float rate=0.005f;
		float dist=glm::clamp(Cam->GetDistance()*(1.0f-dx*rate),0.01f,1000.0f);
		Cam->SetDistance(dist);
	}
}

////////////////////////////////////////////////////////////////////////////////
