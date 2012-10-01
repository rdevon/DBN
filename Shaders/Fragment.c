#version 150
uniform sampler2D weight;
uniform vec4 color;
out vec4 fragColor;
in vec2 tc;
/**
 * Affine transformation
 *
 * Transforms value, which is assumed to be between lower and upper into
 the range specified by lowerOut and upperOut */ 
float affine(float lower, float value, float upper, float lowerOut, float upperOut)
{
   return ((value - lower) / (upper - lower)) * (upperOut - lowerOut) + lowerOut;
}

#extension GL_ARB_separate_shader_objects : enable
void main(void)
{
  fragColor = texture(weight, tc);
  if (fragColor.r <= -100) fragColor = vec4(1,1,1,1);
  else if (fragColor.r <= -99) fragColor = vec4(.5,.5,.5,1);
  else {
     if(fragColor.r < 0.5)
     {
        fragColor = fragColor.r * color;
     }
     else
     {
        float distance = affine(0.5, fragColor.r, 1.0, 0, 1);
        vec3 color = mix(color.rgb, vec3(1,1,0), distance);
        fragColor = vec4(color, 1);
     }
  }
}