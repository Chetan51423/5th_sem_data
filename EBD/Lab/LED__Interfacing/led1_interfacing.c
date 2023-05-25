#include <lpc21xx.h>

void delay(void)
{
	unsigned int i;
	unsigned int j;
	for(i=0; i<100; i++)
	{
		for(j=0; j<100; j++);
	}
	return;
}

int main()
{
	PINSEL0=0x00000000;
	IODIR0 =0x007f8000;
	
	while(1)
	{
		IOSET0=0x007f1000;
		delay();
		IOCLR0=0x007f1000;
		delay();
	}
}