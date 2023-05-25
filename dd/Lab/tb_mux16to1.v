`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   12:55:02 03/31/2023
// Design Name:   mux16to1
// Module Name:   tb_mux16to1.v
// Project Name:  mux16to1
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: mux16to1
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module tb_mux16to1_v;

	// Inputs
	reg [15:0] in;
	reg [3:0] sel;

	// Outputs
	wire out;

	// Instantiate the Unit Under Test (UUT)
	mux16to1 uut (
		.in(in), 
		.sel(sel), 
		.out(out)
	);

	initial begin
		// Initialize Inputs
	#5; 	in = 16'h3f0a;       sel = 4'h0;
	#5; 	sel = 4'h1;
	#5; 	sel = 4'h6;
	#5; 	sel = 4'hc;
	

		// Wait 100 ns for global reset to finish
		#10;
        
		// Add stimulus here

	end
      
endmodule

