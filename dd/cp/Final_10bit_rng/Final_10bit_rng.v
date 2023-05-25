`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    00:59:46 04/29/2023 
// Design Name: 
// Module Name:    Final_10bit_rng 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////

module rndm_gen (
    input clock,
    input reset,
    output reg [9:0] rnd     
);                             //this line define module name + has 3 ports clock , reset and rnd

    wire feedback;
    wire [9:0] lfsr_next;   // 2 wires declare here - where we connect different components in a module

    //An LFSR cannot have an all 0 state, thus reset to non-zero value
    reg [9:0] reset_value = 10'h3E7;
    reg [9:0] lfsr;
    reg [3:0] count;

    always @ (posedge clock or posedge reset) begin
        if (reset) begin
            lfsr <= reset_value;
            count <= 4'hF;
            rnd <= 0;
        end
        else begin
            lfsr <= lfsr_next;
            count <= count + 1;
            // a new random value is ready
            if (count == 4'd10) begin
                count <= 0;
                rnd <= lfsr; //assign the lfsr number to output after 10 shifts
            end
        end
    end

    // X10+x7+x3+x2+1
    assign feedback = lfsr[9] ^ lfsr[6] ^ lfsr[2] ^ lfsr[1] ^ 1'b1;
    assign lfsr_next = {lfsr[8:0], feedback};

endmodule
