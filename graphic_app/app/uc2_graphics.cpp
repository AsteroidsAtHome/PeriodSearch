// This file is part of BOINC.
// http://boinc.berkeley.edu
// Copyright (C) 2012 University of California
//
// BOINC is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// BOINC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with BOINC.  If not, see <http://www.gnu.org/licenses/>.

// Example graphics application, paired with uc2.cpp
// This demonstrates:
// - using shared memory to communicate with the worker app
// - reading XML preferences by which users can customize graphics
//   (in this case, select colors)
// - handle mouse input (in this case, to zoom and rotate)
// - draw text and 3D objects using OpenGL
//
// - Expects TrueType font 0 (by default, LiberationSans-Regular.ttf) 
//   to be in the current directory.  
// - Must be linked with api/ttfont.cpp, libfreetype.a and libftgl.a.
//   (libfreetype.a may also require linking with -lz and -lbz2.)
//   See comments at top of api/ttfont.cpp for more information.
//

//     ---    Asteroids@home    ---
// Graphic application that illustrates result data of calculation process of period search
// It will also point to the current best 'rms' along with the best Period in JD
//
// Developed by Georgi Vidinski (c) 2021
// E-mail: gvidinski@hotmail.co.uk
//

#ifdef _WIN32
#include "boinc_win.h"
#else
#include <cmath>
#endif

#include <filesystem>
#include <chrono>
#include "parse.h"
#include "util.h"
#include <gl/glew.h>
#include "gutil.h"
#include "boinc_gl.h"
#include <GLFW/glfw3.h>
#include "app_ipc.h"
#include "boinc_api.h"
#include "graphics2.h"
#include "ttfont.h"
#include "uc2.h"
#include "diagnostics.h"
#include "curl.h"
#include <nlohmann/json.hpp>
#include "PeriodStruct.h"
#include "SdbdStructs.h"
#include "Helpers.h"
#include "PeriodChart.h"
#include "shader.hpp"
#include "glut.h"

#include "res_texture.c"
#include "../common/shader_utils.h"



#ifdef __APPLE__
#include "mac/app_icon.h"
#endif

using TTFont::ttf_render_string;
using TTFont::ttf_load_fonts;
using nlohmann::json;
using namespace std;
using namespace std::chrono;
using namespace sdbd;
using namespace ps;

const char project_name[] = "asteroidsathome";
string wu_name;

float white[4] = { 1., 1., 1., 1. };
TEXTURE_DESC logo;
int width, height;      // window dimensions
APP_INIT_DATA uc_aid;
bool mouse_down = false;
int mouse_x, mouse_y;
double pitch_angle, roll_angle, viewpoint_distance = 10;
float color[4] = { .7, .2, .5, 1 };
// the color of the 3D object.
// Can be changed using preferences
UC_SHMEM* shmem = NULL;

const string soft_link_in = "period_search_in";
const string soft_link_out = "period_search_out";
const string wu_out_temp_filename = "test_out_temp";
double last_file_size = 0.0;

string asteroid_designator;
string sbdb_json;

string sbdb_api_url = "https://ssd-api.jpl.nasa.gov/sbdb.api?sstr=";
static string gs_strLastResponse;
Asteroid asteroid;


int dataCheckPeriod = 2000; // We want to check the 'period_out' file on every 2 seconds;
steady_clock::time_point lastMillis;

PeriodStruct Ps;
double best_period = 0.0;
int best_period_lambda = 0;
int best_period_beta = 0;
int period_count = 0;

// Create a vertex array object
GLFWwindow* window;
GLuint program;
GLint attribute_coord2d;
GLint uniform_offset_x;
GLint uniform_scale_x;
GLint uniform_sprite;
GLuint texture_id;
GLint uniform_mytexture;

float offset_x = 0.0;
float scale_x = 1.0;
int mode = 0;

struct point {
	GLfloat x;
	GLfloat y;
};

struct Range
{
	double min;
	double max;
};

GLuint vbo;

typedef struct
{
	GLfloat x, y, z;
	GLfloat r, g, b, a;
} Vertex;

vector<point> visual;
int dot_i = 0;
double min_rms = 0.0;
point marker;
float marker_step = 0.0f;
Range range{}; // Holds min/max rms range;
PROGRESS_2D progress;
double fd = 0.0;

// set up lighting model
static void init_lights() {
	GLfloat ambient[] = { 1., 1., 1., 1.0 };
	GLfloat position[] = { -13.0, 6.0, 20.0, 1.0 };
	GLfloat dir[] = { -1, -.5, -3, 1.0 };
	glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
	glLightfv(GL_LIGHT0, GL_POSITION, position);
	glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, dir);
}

static void draw_logo() {
	if (logo.present) {
		//float pos[3] = { .1f, .1f, .0f };
		//float size[3] = { .8f, .6f, .0f };
		float pos[3] = { 0.15f, 0.1f, .0f };
		float size[3] = { 0.8f, 0.6f, .0f };
		logo.draw(pos, size, ALIGN_CENTER, ALIGN_CENTER);
	}
}

static void draw_text() {
	static double x = -0.1, y = 0;
	//static double dx = 0.0003, dy = 0.0007;
	char buf[256];
	//x += dx;
	//y += dy;
	//if (x < 0 || x > .5) dx *= -1;
	//if (y < 0 || y > .4) dy *= -1;
	//double fd = 0;
	double cpu = 0, dt;
	if (shmem) {
		fd = shmem->fraction_done;
		cpu = shmem->cpu_time;
	}
	sprintf(buf, "User: %s", uc_aid.user_name);
	ttf_render_string(x, y + 0.1, 0, 1250, white, buf, 12);
	sprintf(buf, "Team: %s", uc_aid.team_name);
	ttf_render_string(x, y + 0.075, 0, 1250, white, buf, 12);
	sprintf(buf, "Done: %.4f %%", 100 * fd);
	ttf_render_string(x, y + 0.05, 0, 1250, white, buf, 12);
	/*sprintf(buf, "CPU time: %f", cpu);
	ttf_render_string(x, y + 0.125, 0, 1250, white, buf, 12);*/
	if (shmem) {
		dt = dtime() - shmem->update_time;
		if (dt > 10) {
			boinc_close_window_and_quit("shmem not updated");
		}
		else if (dt > 5) {
			ttf_render_string(x, y + 0.025, 0, 1250, white, "App not running - exiting in 5 seconds", 12);
		}
		else if (shmem->status.suspended) {
			ttf_render_string(x, y + 0.025, 0, 1250, white, "App suspended", 12);
		}
	}
	else {
		ttf_render_string(x, y + 0.025, 0, 1250, white, "No shared mem", 12);
	}
	sprintf(buf, "Asteroid: %s", asteroid.object.fullname.c_str());
	ttf_render_string(x, .7, 0, 1000, white, buf, 12);
	sprintf(buf, "Object class: %s", asteroid.object.orbit_class.name.c_str());
	ttf_render_string(x, .67, 0, 1250, white, buf, 12);
	sprintf(buf, "MOID relative to Earth (au): %.2f", asteroid.orbit.moid);
	ttf_render_string(x, .64, 0, 1250, white, buf, 12);
	sprintf(buf, "MOID relative to Jupiter (au): %.2f", asteroid.orbit.moid_jup);
	ttf_render_string(x, .61, 0, 1250, white, buf, 12);
	sprintf(buf, "Best period in JD: %.6f out of %d", best_period, period_count);
	ttf_render_string(x, .58, 0, 1250, white, buf, 12);

	sprintf(buf, "Lambda: %d, Beta: %d", best_period_lambda, best_period_beta);
	ttf_render_string(x, .55, 0, 1250, white, buf, 12);

#ifdef DEBUG
	sprintf(buf, "w: %d, h: %d", width, height);
	ttf_render_string(x, .50, 0, 1250, white, buf, 12);
#endif
}

static void draw_3d_stuff() {
	static float x = 0, y = 0, z = 10;
	static float dx = 0.3, dy = 0.2, dz = 0.5;
	x += dx;
	y += dy;
	z += dz;
	if (x < -15 || x > 15) dx *= -1;
	if (y < -15 || y > 15) dy *= -1;
	if (z < 0 || z > 40) dz *= -1;
	float pos[3];
	pos[0] = x;
	pos[1] = y;
	pos[2] = z;
	drawSphere(pos, 4);
	drawCylinder(false, pos, 6, 6);
}

int init_resources() {
	program = create_program("graph.v.glsl", "graph.f.glsl");
	if (program == 0)
		return 0;

	attribute_coord2d = get_attrib(program, "coord2d");
	uniform_offset_x = get_uniform(program, "offset_x");
	uniform_scale_x = get_uniform(program, "scale_x");
	uniform_sprite = get_uniform(program, "sprite");
	uniform_mytexture = get_uniform(program, "mytexture");

	if (attribute_coord2d == -1 || uniform_offset_x == -1 || uniform_scale_x == -1 || uniform_sprite == -1 || uniform_mytexture == -1)
		return 0;

	/* Enable blending */
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	/* Enable point sprites (not necessary for true OpenGL ES 2.0) */
#ifndef GL_ES_VERSION_2_0
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
#endif

	/* Upload the texture for our point sprites */
	glActiveTexture(GL_TEXTURE0);
	glGenTextures(1, &texture_id);
	glBindTexture(GL_TEXTURE_2D, texture_id);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res_texture.width, res_texture.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, res_texture.pixel_data);

	// Create the vertex buffer object
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	// Create our own temporary buffer
	point graph[2000];

	// Fill it in just like an array
	for (int i = 0; i < 2000; i++) {
		float x = (i - 1000.0) / 100.0;

		graph[i].x = x;
		graph[i].y = sin(x * 10.0) / (1.0 + x * x);
	}

	// Tell OpenGL to copy our array to the buffer object
	glBufferData(GL_ARRAY_BUFFER, sizeof graph, graph, GL_STATIC_DRAW);

	return 1;
}

void display() {

	glUseProgram(program);
	glUniform1i(uniform_mytexture, 0);

	glUniform1f(uniform_offset_x, offset_x);
	glUniform1f(uniform_scale_x, scale_x);

	//glClearColor(0.0, 0.0, 0.0, 0.0);
	//glClear(GL_COLOR_BUFFER_BIT);

	/* Draw using the vertices in our vertex buffer object */
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	glEnableVertexAttribArray(attribute_coord2d);
	glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);

	/* Push each element in buffer_vertices to the vertex shader */
	switch (mode) {
	case 0:
		glUniform1f(uniform_sprite, 0);
		glDrawArrays(GL_LINE_STRIP, 0, 2000);
		break;
	case 1:
		glUniform1f(uniform_sprite, 1);
		glDrawArrays(GL_POINTS, 0, 2000);
		break;
	case 2:
		glUniform1f(uniform_sprite, res_texture.width);
		glDrawArrays(GL_POINTS, 0, 2000);
		break;
	}

	//glutSwapBuffers();
}

void set_viewpoint(double dist) {
	double x, y, z;
	x = 0;
	y = 3.0 * dist;
	z = 11.0 * dist;
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(
		x, y, z,        // eye position
		0, -.8, 0,        // where we're looking
		0.0, 1.0, 0.    // up is in positive Y direction
	);
	glRotated(pitch_angle, 1., 0., 0);
	glRotated(roll_angle, 0., 1., 0);
}

static void init_camera(double dist) {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(
		45.0,       // field of view in degree
		1.0,        // aspect ratio
		1.0,        // Z near clip
		1000.0      // Z far
	);
	set_viewpoint(dist);
}

void drawPoint(Vertex v1, GLfloat size)
{
	glPointSize(size);
	glBegin(GL_POINTS);
	glColor4f(v1.r, v1.g, v1.b, v1.a);
	glVertex3f(v1.x, v1.y, v1.z);
	glEnd();
}

double MapValue(double x, double in_min, double in_max, double out_min, double out_max) {
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

void PrepareChartRms(const double scale = 1)
{
	if (Ps.rms.empty())
	{
		return;
	}

	const auto dataSize = static_cast<size_t>(Ps.rms.size());
	const auto step = dataSize > 100
		? (1.0 - 0.2) / static_cast<double>(dataSize)
		: 0.008;

	auto rmsIter = std::min_element(Ps.rms.begin(), Ps.rms.end());
	const auto minIndex = std::distance(Ps.rms.begin(), rmsIter);
	range.min = Ps.rms[minIndex];
	rmsIter = std::max_element(Ps.rms.begin(), Ps.rms.end());
	const auto maxIndex = std::distance(Ps.rms.begin(), rmsIter);
	range.max = Ps.rms[maxIndex];
	//const auto rmsDelta = range.max - range.min;
	/*const auto yStep = (0.5 - 0.1) / 3;
	auto yOffset = rmsDelta * yStep * 2;*/
	visual.resize(dataSize);

	for (size_t i = 0; i < dataSize; i++)
	{
		visual[i].x = 0.2 + float(i * step);
		visual[i].y = MapValue(Ps.rms[i], range.min, range.max, 0.1, 0.4);
	}

	min_rms = range.min;
	marker.x = visual[minIndex].x;
	marker.y = visual[minIndex].y;
}

void DrawChartRms()
{
	const auto dataSize = static_cast<size_t>(Ps.rms.size());
	if (dataSize == 0) return;

	//set the width of the line
	glLineWidth(1.0f);
	glBegin(GL_LINE_STRIP);

	//set the color of the line to green
	glColor4f(0.1f, 1.0f, 0.1f, 1.0f);

	if (dot_i == dataSize)
	{
		// TODO: Add waiter for 5 seconds
		dot_i = 0;
	}

	//auto t = dataSize - dot_i;

	for (size_t i = 0; i < dataSize; i++)
	{
		glVertex3f(visual[i].x, 0.1f + visual[i].y, 1.0f);
	}


	glEnd();

	/*const Vertex dotTmp = { visual[dot_i].x, 0.1f + visual[dot_i].y, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
	drawPoint(dotTmp, 5);

	dot_i++;*/

	/*glLineWidth(1.0f);
	glBegin(GL_LINES);
	glColor4f(0.5f, 0.5f, 1.0f, 0.8f);
	for (size_t i = 1; i < dataSize; i++)
	{
		glVertex3f(v[i].x, 0.2f + v[i].y - 0.005f, 1.0f);
		glVertex3f(v[i].x, 0.2f + v[i].y + 0.005f, 1.0f);
	}

	glEnd();*/
}

void DrawSpinner()
{
	if (!min_rms > 0) return;
	if (marker_step == 90) marker_step = 0;
	for (size_t j = 0; j < 4; j++)
	{
		for (size_t i = 10 + marker_step; i < 80 + marker_step; i++)
			//for (size_t i = 135; i > 45; i--)
		{
			auto smallX = cos(2 * 3.14159 * (i + (j * 90)) / 360.0) * -1;
			auto x = marker.x + smallX / 50;
			auto y = 0.1 + marker.y + (sin(2 * 3.14159 * (i + (j * 90)) / 360.0) / 50);
			//glColor4f(0.5f, 0.5, 1.0f, 1.0f);
			//glVertex3f(x, y, 0);

			//Vertex spb = { x, y, 0.0f, 1.0f, 1.0f, 1.0f, 0.5f };
			//drawPoint(spb, 3);

			Vertex sp = { x, y, 0.1f, 0.0f, 0.2f, 1.0f, 0.8f };
			drawPoint(sp, 2.0f);
		}

	}

	auto alpha = 4 * 2 * 3.14159 * marker_step / 360;
	GLfloat size = (cos(alpha) + 1) * 5;
	//cout << marker_step << " | " << alpha << " | " << size << " | " << (size + 1) * 5 << endl;
	Vertex p = { marker.x, 0.1f + marker.y, 0.1f, 1.0f, 1.0f, 1.0f, 0.8f };
	drawPoint(p, size);

	marker_step += 1;
}

void GetMaxPeriod()
{
	//if (GetWorkUnitOutputData(soft_link_out, wu_out_temp_filename)) return;
	if (GetPeriodOutputData(soft_link_out, last_file_size, Ps)) return;

	// -----------------------------------
	// NOTE: To save us from troubles we will follow the "BOINC API paper" and will not going to use the fancy c++ approach
	// More about the case here: https://boinc.berkeley.edu/boinc_papers/api/text.html
	// 6. Directory structure and file access
	// -----------------------------------

	//ifstream ifs;
	//ofstream wu_out_temp;
	//ifs.open(wu_out_filename, ios::in | ios::binary);
	//wu_out_temp.open(wu_out_temp_filename, ios::out | ios::binary);
	////now = steady_clock::now();
	//
	//wu_out_temp << ifs.rdbuf();

	////last = steady_clock::now();
	////duration = duration_cast<milliseconds>(last - now).count();
	////cout << "Duration: " << duration << " ms" << endl;

	//ifs.close();
	//wu_out_temp.close();
	// -----------------------------------

	auto minRmsIter = std::min_element(Ps.rms.begin(), Ps.rms.end());
	auto index = std::distance(Ps.rms.begin(), minRmsIter);
	//auto minRms = *minRmsIter;

	// Get Period and other values from vectors by index
	best_period = Ps.period[index];
	best_period_lambda = Ps.alpha[index];
	best_period_beta = Ps.beta[index];
	period_count = Ps.period.size();

	//cout << bestPeriod << endl;
}

void GetData()
{
	// NOTE: We will read data from the file on every 'dataCheckPeriod' (f.e. 2000ms)
	const auto now = steady_clock::now();
	const auto duration = duration_cast<milliseconds>(now - lastMillis).count();
	if (duration < dataCheckPeriod)
	{
		return;
	}

	lastMillis = now;

	// while testing
	//auto duration_in_seconds = std::chrono::duration<double>(now.time_since_epoch());
	//int num_seconds = duration_in_seconds.count();

	//std::cout << num_seconds << endl;

	GetMaxPeriod();
	PrepareChartRms();
}

void drawPointsDemo(int width, int height) {
	GLfloat size = 5.0f;
	for (auto i = 0; i <= 10; i += 2, size += 5)
	{
		const GLfloat x = static_cast<float>(i) / 10;
		const Vertex v1 = { x, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };
		drawPoint(v1, size);
	}
}

void drawLineSegment(Vertex v1, Vertex v2, GLfloat width) {
	glLineWidth(width);
	glBegin(GL_LINES);
	glColor4f(v1.r, v1.g, v1.b, v1.a);
	glVertex3f(v1.x, v1.y, v1.z);
	glColor4f(v2.r, v2.g, v2.b, v2.a);
	glVertex3f(v2.x, v2.y, v2.z);
	glEnd();
}

void InitProgress()
{
	GLfloat position[] = { 0.2f, 0.06f, 0.1f };
	const auto alpha = 1.0f;
	GLfloat color[] = { 0.5f, 0.4f, 0.1f, alpha / 2 };
	GLfloat innerColor[] = { 0.1f, 0.8f, 0.2f, alpha };

	progress.init(position, 0.8f, .01f, 0.008f, color, innerColor);
}

void DrawProgress()
{
	progress.draw(static_cast<float>(fd));
}

void DrawCoordinateSystem()
{
	// Ordinate
	/*Vertex v1 = { 0.3f, 0.1f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f };
	Vertex v2 = { 0.3f, 0.5f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f };
	drawLineSegment(v1, v2, 1);*/

	//Abscissa
	Vertex v1 = { 0.2f, 0.1f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f };
	Vertex v2 = { 1.0f, 0.1f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f };
	drawLineSegment(v1, v2, 1);

	v1 = { 0.56f, 0.13f, 1.0f, 0.1f, 1.0f, 0.1f, 1.0f };
	v2 = { 0.58f, 0.13f, 1.0f, 0.1f, 1.0f, 0.1f, 1.0f };
	drawLineSegment(v1, v2, 1);

	ttf_render_string(0.6, 0.125, 0, 1250, white, "rms", 12);
}

void DrawMiddleSectors()
{
	char buf[256];
	//auto diff = range.max - range.min;
	//auto factor = 1;
	//auto tempDiff = 0.0;
	//if (diff != 0)
	//{
	//	while (tempDiff < 1)
	//	{
	//		tempDiff = diff;
	//		factor *= 10;
	//		tempDiff *= factor;
	//	}

	//	diff = round(tempDiff) / factor;
	//}

	//auto y = static_cast<GLfloat>(diff);

	Vertex v1 = { 0.2f, 0.2f, 1.0f, 0.0f, 0.5f, 0.5f, 1.0f };
	Vertex v2 = { 1.0f, 0.2f, 1.0f, 0.0f, 0.5f, 0.5f, 1.0f };
	drawLineSegment(v1, v2, 0.5);

	sprintf(buf, "%.4f", range.min);
	ttf_render_string(0.2, .21, 0, 1250, white, buf, 12);

	v1 = { 0.2f, 0.35f, 1.0f, 0.0f, 0.5f, 0.5f, 1.0f };
	v2 = { 1.0f, 0.35f, 1.0f, 0.0f, 0.5f, 0.5f, 1.0f };
	drawLineSegment(v1, v2, 0.5);

	v1 = { 0.2f, 0.5f, 1.0f, 0.0f, 0.5f, 0.5f, 1.0f };
	v2 = { 1.0f, 0.5f, 1.0f, 0.0f, 0.5f, 0.5f, 1.0f };
	drawLineSegment(v1, v2, 0.5);
}

void ShootingStar()
{
	//visual
	auto dataSize = static_cast<size_t>(visual.size());
	if (dataSize == 0) return;
	if (dot_i == dataSize)
	{
		dot_i = 0;
	}

	Vertex v1 = { 0.2f, 0.1f + visual[dot_i].y, 1.0f, 0.0f, 0.5f, 0.5f, 1.0f };
	Vertex v2 = { 1.0f, 0.1f + visual[dot_i].y, 1.0f, 0.0f, 0.5f, 0.5f, 1.0f };
	drawLineSegment(v1, v2, 0.5);

	v1 = { visual[dot_i].x, 0.1f, 1.0f, 0.0f, 0.5f, 0.5f, 1.0f };
	v2 = { visual[dot_i].x, 0.5f, 1.0f, 0.0f, 0.5f, 0.5f, 1.0f };
	drawLineSegment(v1, v2, 0.5);

	const Vertex dotTmp = { visual[dot_i].x, 0.1f + visual[dot_i].y, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };


	drawPoint(dotTmp, 5);
	dot_i++;
}

void SomeDraw()
{
	float ratio = (float)width / (float)height;

	//glViewport(0, 0, width, height);
	//glClear(GL_COLOR_BUFFER_BIT);
	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	////Orthographic Projection
	//glOrtho(-ratio, ratio, -1.f, 1.f, 1.f, -1.f);
	//glMatrixMode(GL_MODELVIEW);
	//glLoadIdentity();
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//drawPointsDemo(width, height);
	DrawCoordinateSystem();
	DrawMiddleSectors();
	DrawChartRms();
	DrawSpinner();
	DrawProgress();
	//ShootingStar();
}

void SetAntialias()
{
	glEnable(GL_POINT_SMOOTH);
	glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void app_graphics_render(int xs, int ys, double time_of_day) {
	// boinc_graphics_get_shmem() must be called after 
	// boinc_parse_init_data_file()
	// Put this in the main loop to allow retries if the 
	// worker application has not yet created shared memory
	//
	if (shmem == NULL) {
		char shmemName[256];
		snprintf(shmemName, sizeof shmemName, "%s_%s", project_name, wu_name.c_str());
		shmem = (UC_SHMEM*)boinc_graphics_get_shmem(shmemName);
	}
	if (shmem) {
		shmem->countdown = 5;
	}

	SetAntialias();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// draw logo first - it's in background
	//
	mode_unshaded();
	mode_ortho();
	draw_logo();
	ortho_done();

	GetData();

	// draw 3D objects
	//
	init_camera(viewpoint_distance);
	scale_screen(width, height);
	mode_shaded(color);
	//draw_3d_stuff();

	// draw text on top
	//
	mode_unshaded();
	mode_ortho();
	draw_text();
	ortho_done();

	//InitLine();
	mode_unshaded();
	mode_ortho();
	//RenderLine();

	SomeDraw();
	//drawShape();

	ortho_done();

	//display();
	//DisplayChart(vao, window);
}

void app_graphics_resize(int w, int h) {
	width = w;
	height = h;
	glViewport(0, 0, w, h);
}

// mouse drag w/ left button rotates 3D objects;
// mouse draw w/ right button zooms 3D objects
//
void boinc_app_mouse_move(int x, int y, int left, int middle, int right) {
	if (left) {
		pitch_angle += (y - mouse_y) * .1;
		roll_angle += (x - mouse_x) * .1;
		mouse_y = y;
		mouse_x = x;
	}
	else if (right) {
		double d = (y - mouse_y);
		viewpoint_distance *= exp(d / 100.);
		mouse_y = y;
		mouse_x = x;
	}
	else {
		mouse_down = false;
	}
}

void boinc_app_mouse_button(int x, int y, int which, int is_down) {
	if (is_down) {
		mouse_down = true;
		mouse_x = x;
		mouse_y = y;
	}
	else {
		mouse_down = false;
	}
}

void boinc_app_key_press(int, int) {}

void boinc_app_key_release(int, int) {}


void GetAsteroidDesignator()
{
	wu_name = GetFileNameFromSoftLink(soft_link_in);
	//APP_INIT_DATA aid;
	//int retval = boinc_get_init_data(aid);
	//const auto wuName = aid.wu_name;
	//const string filename(wuName);

	const auto s1 = wu_name.substr(wu_name.find("ps_") + 3);
	asteroid_designator = s1.substr(0, s1.find("_input"));
	sbdb_api_url += asteroid_designator;
}

size_t FunctionPt(void* ptr, size_t size, size_t nmemb, void* /*stream*/)
{
	gs_strLastResponse += static_cast<const char*>(ptr);
	return size * nmemb;
}

bool CallServerWithCurl(std::string& strErrorDescription)
{
	auto* const curl = curl_easy_init();
	if (curl == nullptr)
	{
		strErrorDescription = "Unable to initialise Curl";
		return false;
	}

	curl_easy_setopt(curl, CURLOPT_URL, sbdb_api_url.c_str());
	curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
	//curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);     // enable verbose for easier tracing

	gs_strLastResponse = "";
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, FunctionPt);        // set a callback to capture the server's response

	const auto res = curl_easy_perform(curl);
	if (res != CURLE_OK)
	{
		strErrorDescription = "Curl call to server failed";
		return false;
	}

	/*if (!DoSomethingWithServerResponse(gs_strLastResponse))
	{
		strErrorDescription = "Curl call to server returned an unexpected response";
		return false;
	}*/

	// extract some transfer info
	double speed_upload, total_time;
	curl_easy_getinfo(curl, CURLINFO_SPEED_DOWNLOAD, &speed_upload);
	curl_easy_getinfo(curl, CURLINFO_TOTAL_TIME, &total_time);
	fprintf(stderr, "Requesting data from \"%s\"\n", sbdb_api_url.c_str());
	fprintf(stderr, "At speed: %.3f bytes/sec during %.3f seconds\n\n", speed_upload, total_time);

	curl_easy_cleanup(curl);

	return true;
}

void InitGlfw()
{
	// Initialize GLFW
	glewExperimental = GL_TRUE;
	auto err = glewInit();
	if (GLEW_OK != err)
	{
		/* Problem: glewInit failed, something is seriously wrong. */
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		exit(-1);
	}

	/*if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW! I'm out!" << std::endl;
		exit(-1);
	}*/

	window = glfwCreateWindow(width, height, "Test", NULL, NULL);
	glfwMakeContextCurrent(window);

	//InitializeChart(vao);
}

void ReadAsteroidData()
{
	// read a JSON file
	auto rdbdFileName = asteroid_designator + "_rdbd.json";
	json jsonRdbd;
	std::ifstream rdbdFile(rdbdFileName);
	if (rdbdFile.fail()) {
		//File does not exist code here
		std::string strErrorDescription;
		CallServerWithCurl(strErrorDescription);
		jsonRdbd = json::parse(gs_strLastResponse);
		std::ofstream outRdbdFile(rdbdFileName);
		outRdbdFile << std::setw(4) << jsonRdbd << std::endl;
	}
	else
	{
		rdbdFile >> jsonRdbd;
		rdbdFile.close();
	}

	asteroid = jsonRdbd.get<sdbd::Asteroid>();

	lastMillis = steady_clock::now();
}

void GetAsteroidData()
{
	GetAsteroidDesignator();
	ReadAsteroidData();
}

// Init 
void app_graphics_init()
{
	char path[256];

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

	ttf_load_fonts(strlen(uc_aid.project_dir) ? uc_aid.project_dir : "ttf");

	//boinc_resolve_filename("asteroid_560x354.jpg", path, sizeof(path));
	boinc_resolve_filename("asteroid_800_450.jpg", path, sizeof(path));
	logo.load_image_file(path);

	init_lights();

	InitProgress();

	GetAsteroidData();
}

static void parse_project_prefs(char* buf) {
	char cs[256];
	COLOR c;
	double hue;
	double max_frames_sec, max_gfx_cpu_pct;
	if (!buf) return;
	if (parse_str(buf, "<color_scheme>", cs, 256)) {
		if (!strcmp(cs, "Tahiti Sunset")) {
			hue = .9;
		}
		else if (!strcmp(cs, "Desert Sands")) {
			hue = .1;
		}
		else {
			hue = .5;
		}
		HLStoRGB(hue, .5, .5, c);
		color[0] = c.r;
		color[1] = c.g;
		color[2] = c.b;
		color[3] = 1;
	}
	if (parse_double(buf, "<max_frames_sec>", max_frames_sec)) {
		boinc_max_fps = max_frames_sec;
	}
	if (parse_double(buf, "<max_gfx_cpu_pct>", max_gfx_cpu_pct)) {
		boinc_max_gfx_cpu_frac = max_gfx_cpu_pct / 100;
	}
}

int main(int argc, char** argv) {
	//boinc_init_graphics_diagnostics(BOINC_DIAG_DEFAULTS);
	boinc_init_graphics_diagnostics(BOINC_DIAG_MEMORYLEAKCHECKENABLED | \
		BOINC_DIAG_REDIRECTSTDERR);

#ifdef __APPLE__
	setMacIcon(argv[0], MacAppIconData, sizeof(MacAppIconData));
#endif

	boinc_parse_init_data_file();
	boinc_get_init_data(uc_aid);
	if (uc_aid.project_preferences) {
		parse_project_prefs(uc_aid.project_preferences);
	}
	
	boinc_graphics_loop(argc, argv);

	boinc_finish_diag();
}
