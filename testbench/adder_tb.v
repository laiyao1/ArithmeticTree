`timescale 1ns / 1ps

module adder_tb();

// Inputs
reg [63:0] a;
reg [63:0] b;

// Outputs
wire [63:0] s;
wire cout;

// Expected output
reg [64:0] expected;

// Instantiate the Unit Under Test (UUT)
main uut (
    .a(a), 
    .b(b), 
    .s(s), 
    .cout(cout)
);

integer i;

initial begin
    // Initialize Inputs
    a = 0;
    b = 0;

    // Wait 100 ns for global reset to finish
    #100;
    
    // Test cases
    for (i = 0; i < 100; i = i + 1) begin
        case (i)
            // Edge cases
            0: begin a = 64'h0000000000000000; b = 64'h0000000000000000; end
            1: begin a = 64'hFFFFFFFFFFFFFFFF; b = 64'h0000000000000001; end
            2: begin a = 64'hFFFFFFFFFFFFFFFF; b = 64'hFFFFFFFFFFFFFFFF; end
            3: begin a = 64'h8000000000000000; b = 64'h8000000000000000; end
            4: begin a = 64'hFFFFFFFFFFFFFFFE; b = 64'h0000000000000001; end
            
            // Powers of 2
            5: begin a = 64'h0000000000000001; b = 64'h0000000000000001; end
            6: begin a = 64'h0000000000000002; b = 64'h0000000000000002; end
            7: begin a = 64'h0000000000000004; b = 64'h0000000000000004; end
            8: begin a = 64'h0000000000000008; b = 64'h0000000000000008; end
            9: begin a = 64'h0000000000000010; b = 64'h0000000000000010; end
            
            // Alternating bit patterns
            10: begin a = 64'h5555555555555555; b = 64'hAAAAAAAAAAAAAAAA; end
            11: begin a = 64'hAAAAAAAAAAAAAAAA; b = 64'h5555555555555555; end
            
            // Random numbers (using $random for simplicity, but ensuring they're positive)
            default: begin
                a = {$random, $random} & 64'h7FFFFFFFFFFFFFFF; // Ensure positive
                b = {$random, $random} & 64'h7FFFFFFFFFFFFFFF; // Ensure positive
            end
        endcase

        expected = a + b;
        #10;
        if ({cout, s} == expected)
            $display("Test case %d passed: %h + %h = %h", i, a, b, {cout, s});
        else
            $display("Test case %d failed. %h + %h: Expected %h, got %h", i, a, b, expected, {cout, s});
    end

    $finish;
end

// Optional: Dump waveforms
initial begin
    $dumpfile("adder_tb.vcd");
    $dumpvars(0, adder_tb);
end

endmodule