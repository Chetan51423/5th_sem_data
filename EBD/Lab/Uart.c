#include<lpc21xx.h>

int getkey(void);

char mystring[4]={'V','I','T',0x0A};

void delay()

{ int j=0;

	for(j=0;j<1000000;j++);

}

int i=0;

int main()

{

/* UART Initialisation */

U0FCR=0X01;
//ENABLE UART0

PINSEL0=0X000005;
//Enable Txd and Rxd UART0

U0LCR=0X83;
//DLAB=1;1 STOP BIT; 8-BIT DATA

U0DLL=97;
//Baud Rate=9600bps @ VPB clock=15MHz.

U0LCR=0X03;
//DLAB=0 for serial communication

U0THR=0;

/* Transmission */

while(1)

{
	for(i=0;i<4;i++)
	{
       while(!(U0LSR&(0X20)));

       U0THR=mystring[i];

       delay();
    }
}
}