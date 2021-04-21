#include <cstdio>
//#include "boinc_gl.h"
#include "gl/glew.h"
//#include "boinc_glut.h"
//#include "gl/GL.h"
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
//#include "glfw3native.h"
#include <GLFW/glfw3native.h>

#include "shader.hpp"
#include <cmath>

#include "res_texture.c"
#include "../common/shader_utils.h"

//GLuint program;

//GLint attribute_coord2d;
//GLint uniform_offset_x;
//GLint uniform_scale_x;
//GLint uniform_sprite;
//GLuint texture_id;
//GLint uniform_mytexture;

//int mode = 0;

struct point {
	GLfloat x;
	GLfloat y;
};

GLuint VertexShaderId,
	FragmentShaderId,
	ProgramId,
	VaoId,
	VboId,
	ColorBufferId;


//int init_resources(GLuint &vbo) {
//	program = LoadShaders("graph.v.glsl", "graph.f.glsl");
//	if (program == 0)
//		return 0;
//
//	attribute_coord2d = get_attrib(program, "coord2d");
//	uniform_offset_x = get_uniform(program, "offset_x");
//	uniform_scale_x = get_uniform(program, "scale_x");
//	uniform_sprite = get_uniform(program, "sprite");
//	uniform_mytexture = get_uniform(program, "mytexture");
//
//	if (attribute_coord2d == -1 || uniform_offset_x == -1 || uniform_scale_x == -1 || uniform_sprite == -1 || uniform_mytexture == -1)
//		return 0;
//
//	/* Enable blending */
//	glEnable(GL_BLEND);
//	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//
//	/* Enable point sprites (not necessary for true OpenGL ES 2.0) */
//#ifndef GL_ES_VERSION_2_0
//	glEnable(GL_POINT_SPRITE);
//	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
//#endif
//
//	/* Upload the texture for our point sprites */
//	glActiveTexture(GL_TEXTURE0);
//	glGenTextures(1, &texture_id);
//	glBindTexture(GL_TEXTURE_2D, texture_id);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res_texture.width, res_texture.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, res_texture.pixel_data);
//
//	// Create the vertex buffer object
//	glGenBuffers(1, &vbo);
//	glBindBuffer(GL_ARRAY_BUFFER, vbo);
//
//	// Create our own temporary buffer
//	point graph[2000];
//
//	// Fill it in just like an array
//	for (int i = 0; i < 2000; i++) {
//		float x = (i - 1000.0) / 100.0;
//
//		graph[i].x = x;
//		graph[i].y = sin(x * 10.0) / (1.0 + x * x);
//	}
//
//	// Tell OpenGL to copy our array to the buffer object
//	glBufferData(GL_ARRAY_BUFFER, sizeof graph, graph, GL_STATIC_DRAW);
//
//	//// Get the location of the attributes that enters in the vertex shader
//	//GLint position_attribute = glGetAttribLocation(program, "position");
//
//	//// Specify how the data for position can be accessed
//	//glVertexAttribPointer(position_attribute, 2, GL_FLOAT, GL_FALSE, 0, 0);
//
//	//// Enable the attribute
//	//glEnableVertexAttribArray(position_attribute);
//
//	return 1;
//}

//void display(GLuint& vbo, GLuint& program) {
//
//	glUseProgram(program);
//
//
//	/* Draw using the vertices in our vertex buffer object */
//	glBindBuffer(GL_ARRAY_BUFFER, vbo);
//
//	glEnableVertexAttribArray(attribute_coord2d);
//	glVertexAttribPointer(attribute_coord2d, 2, GL_FLOAT, GL_FALSE, 0, 0);
//
//	/* Push each element in buffer_vertices to the vertex shader */
//	switch (mode) {
//	case 0:
//		glUniform1f(uniform_sprite, 0);
//		glDrawArrays(GL_LINE_STRIP, 0, 2000);
//		break;
//	case 1:
//		glUniform1f(uniform_sprite, 1);
//		glDrawArrays(GL_POINTS, 0, 2000);
//		break;
//	case 2:
//		glUniform1f(uniform_sprite, res_texture.width);
//		glDrawArrays(GL_POINTS, 0, 2000);
//		break;
//	}
//
//
//	
//	/*glClear(GL_COLOR_BUFFER_BIT);
//
//	glBindVertexArray(vao);
//	glDrawArrays(GL_TRIANGLES, 0, 12);*/
//	
//	// Swap front and back buffers
//	//glfwSwapBuffers();
//}

bool InitLine()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
	glMatrixMode(GL_PROJECTION);
	// Set grid to be from 0 to 1
	gluOrtho2D(0.0, 3.0, 0.0, 3.0);

	return true;
}

void drawline(float from_x, float from_y, float to_x, float to_y)
{
	// From coordinate position
	glVertex2f(from_x, from_y);

	// To coordinate position
	glVertex2f(to_x, to_y);
}

void drawPoint(float x, float y)
{
	static const GLfloat g_vertex_buffer_data[] = {
   -1.0f, -1.0f, 0.0f,
   1.0f, -1.0f, 0.0f,
   0.0f,  1.0f, 0.0f,
	};

	// This will identify our vertex buffer
	//GLuint vertexbuffer;
	//// Generate 1 buffer, put the resulting identifier in vertexbuffer
	//glGetBuf
	//glGenBuffers(1, &vertexbuffer);
	//// The following commands will talk about our 'vertexbuffer' buffer
	//glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	//// Give our vertices to OpenGL.
	//glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

	//// 1st attribute buffer : vertices
	//glEnableVertexAttribArray(0);
	//glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	glVertexAttribPointer(
		0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);
	// Draw the triangle !
	glDrawArrays(GL_TRIANGLES, 0, 3); // Starting from vertex 0; 3 vertices total -> 1 triangle
	glDisableVertexAttribArray(0);

	//glDrawArrays(GL_LINE_STRIP, 0, 2000);
}

void drawpoint(int x, int y)
{
	glVertex2f(x, y);
}

void drawShape()
{
	glColor3f(1.0, 1.0, 0.0); // Color (RGB): Yellow
	glLineWidth(1.0); // Set line width to 2.0

	// Draw line
	glBegin(GL_LINES);
	drawline(0.25, 0.5, 0.4, 0.5);
	drawline(0.4, 0.6, 0.4, 0.5);
	drawline(0.4, 0.4, 0.4, 0.5);
	drawline(0.6, 0.5, 0.75, 0.5);
	glEnd();

	glBlendFunc(GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
	glPointSize(1);
	glBegin(GL_LINE_STRIP);
	// Create our own temporary buffer
	point graph[2000];

	// Fill it in just like an array
	for (int i = 0; i < 2000; i++) {
		//float x = (i - 1000.0) / 1000.0;
		float x = (i) / 1000.0;

		graph[i].x = x;
		graph[i].y = 0.5 + (sin(x * 10) / (2.0 + x * x));
		//graph[i].y = sin(x * 10.0) / (1.0 + x * x);
		glVertex2f(graph[i].x, graph[i].y);
	}
	//drawpoint(graph->x, graph->y);
	glEnd();

	// Draw triangle
	/*glBegin(GL_TRIANGLES);
	glVertex2f(0.4, 0.5);
	glVertex2f(0.6, 0.6);
	glVertex2f(0.6, 0.4);
	glEnd();*/

	//drawPoint(0, 0);


}

//int CreateVbo(GLfloat* vertices, int size)
//{
//	GLuint vboId;
//	glGenBuffers(1, &vboId);
//	glBindBuffer(GL_ARRAY_BUFFER, vboId);
//	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
//
//	return vboId;
//}

void RenderLine()
{
	GLfloat Vertices[] = {
		0.1f, 0.5f, -0.05f, 1.0f,
		0.3f, 0.5f, -0.05f, 1.0f,
		0.2f, 0.8f, -0.05f, 1.0f
	};

	GLfloat Colors[] = {
	  1.0f, 0.0f, 0.0f, 1.0f,
	  0.0f, 1.0f, 0.0f, 1.0f,
	  0.0f, 0.0f, 1.0f, 1.0f
	};

	GLenum ErrorCheckValue = glGetError();

	glGenVertexArrays(1, &VaoId);
	glBindVertexArray(VaoId);

	glGenBuffers(1, &VboId);
	glBindBuffer(GL_ARRAY_BUFFER, VboId);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertices), Vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);

	glGenBuffers(1, &ColorBufferId);
	glBindBuffer(GL_ARRAY_BUFFER, ColorBufferId);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Colors), Colors, GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(1);

	ErrorCheckValue = glGetError();
	if (ErrorCheckValue != GL_NO_ERROR)
	{
		fprintf(
			stderr,
			"ERROR: Could not create a VBO: %s \n",
			gluErrorString(ErrorCheckValue)
		);

		exit(-1);
	}
}

void RenderLine_old()
{
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//glOrtho(0.0, 4.0, 0.0, 4.0, -1, 1);
	glOrtho(0, 1, 0, 1, 0, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Draw shape one
	glPushMatrix();
	//glTranslatef(1.5, 1.5, 0.0);
	drawShape();
	glPopMatrix();

	// Draw shape two
	//glPushMatrix();
	//glTranslatef(2.5, 2.5, 0.0);
	//drawShape();
	//glPopMatrix();

	//glutSwapBuffers();

	// ------------
	//glBegin(GL_TRIANGLES);
	//glVertex3f(0.3f, 0.3f, -0.2f);
	//glVertex3f(0.6f, 0.2f, -0.8f);
	//glVertex3f(0.4f, 0.5f, -.5f);
	//glEnd();
	// ------------

	// ------------
	// Enable vertex arrays
	//glEnableClientState(GL_VERTEX_ARRAY);

	//struct vertex
	//{
	//	GLfloat x, y, z;
	//};

	//vertex* vertices = new vertex[3];
	//vertices[0].x = 0.100f;
	//vertices[0].y = 0.050f;
	//vertices[0].z = -0.070f;

	//// Vertex 2
	//vertices[1].x = 0.200f;
	//vertices[1].y = 0.050f;
	//vertices[1].z = -0.010f;

	//// Vertex 3
	//vertices[2].x = 0.150f;
	//vertices[2].y = 0.150f;
	//vertices[2].z = -0.020f;

	//glVertexPointer(3, // number of coordinates per vertex (x,y,z)
	//	GL_FLOAT,       // they are floats
	//	sizeof(vertex), // stride
	//	vertices);      // the array pointer

	//// Render primitives from array-based data
	//int num_indices = 3;
	//glDrawArrays(GL_TRIANGLES, 0, num_indices);
	// -------------
}
