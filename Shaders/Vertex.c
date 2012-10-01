#version 150

in vec4 vertex;
in vec2 texCoord;
uniform mat4 mvp;
out vec2 tc;

void main(void)
{
  gl_Position = mvp * vertex;
  tc = texCoord;
}
