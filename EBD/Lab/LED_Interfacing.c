#include<lpc2148.h>

void delay(void)
{
	unsigned int i;
	for(i=0; i<10000; i++)
	return 0;
}

int main()
{
	PINSEL0=0x00000000;
	IODIR0 =0x007f8000;
	
	while(1)
	{
		IOSET0=0x007f8000;
		delay();
		IOCLR0=0x007f8000;
		delay();
	}
}