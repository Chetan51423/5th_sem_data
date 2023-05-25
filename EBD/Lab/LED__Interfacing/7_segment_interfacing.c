#include <lpc21xx.h>

/*
a= p0.16;
b= p0.17;
c= p0.18;
d= p0.19;
e= p0.20;
f= p0.21;
g= p0.22;
dot= p0.23


DIS1  p0.28
*/



unsigned int delay, count=0, Switchcount=0;
unsigned int Disp[16] = {0x003f0000, 0x00060000, 0x005B0000, 0x004F0000, 0x00660000, 
                         0x006D0000, 0x007D0000, 0x00070000, 0x007F0000, 0x006F0000,
                         0x00770000, 0x007C0000, 0x00390000, 0x005E0000, 0x00790000, 0x00710000};


int main(void)
{
	PINSEL1 = 0x00000000;
	IO0DIR  = 0xF0FF0000;
	
	while(1)
	{
		IO0SET = 0x10000000;
		IO0CLR = 0x00FF0000;
		
	  for(delay =0; delay<100; delay++)
		    IO0SET = Disp[Switchcount];
		for(delay =0; delay<1000000; delay++)
		{}
			
			Switchcount++;
			if(Switchcount == 16)
			{
				Switchcount=0;
			}
	}
}	
												 

