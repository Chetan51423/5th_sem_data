`timescale 1ns / 1ps

////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer:
//
// Create Date:   01:02:40 04/29/2023
// Design Name:   rndm_gen
// Module Name:   rndm_tb.v
// Project Name:  Final_10bit_rng
// Target Device:  
// Tool versions:  
// Description: 
//
// Verilog Test Fixture created by ISE for module: rndm_gen
//
// Dependencies:
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
////////////////////////////////////////////////////////////////////////////////

module rndm_tb;
    wire clock;
    wire reset;
    wire [9:0] rnd;

    reg clk;
    reg rst;

    rndm_gen rando (
        .clock(clock),
        .reset(reset),
        .rnd(rnd)
    );

    assign clock = clk;
    assign reset = rst;

    //generate the clock
    initial begin
        clk <= 0;
        forever begin
            #10;
            clk <= ~clk;
        end
    end

    initial begin
        rst <= 1;
        #50;
        rst <= 0;
    end
      
endmodule


