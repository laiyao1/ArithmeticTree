`timescale 1ns / 1ps

module multiplier_tb();

// Inputs
reg [63:0] x;
reg [63:0] y;

// Outputs
wire [127:0] o;

// Expected output
reg [127:0] expected;

// Instantiate the Unit Under Test (UUT)
main uut (
    .x(x),
    .y(y),
    .o(o)
);

integer i;

initial begin
    // Initialize Inputs
    x = 0;
    y = 0;

    // Wait 100 ns for global reset to finish
    #100;
    
    // Test cases
    for (i = 0; i < 100; i = i + 1) begin
        case (i)
            // Edge cases
            0: begin x = 64'h0000000000000000; y = 64'h0000000000000000; end
            1: begin x = 64'hFFFFFFFFFFFFFFFF; y = 64'h0000000000000001; end
            2: begin x = 64'hFFFFFFFFFFFFFFFF; y = 64'hFFFFFFFFFFFFFFFF; end
            3: begin x = 64'h8000000000000000; y = 64'h0000000000000002; end
            4: begin x = 64'hFFFFFFFFFFFFFFFE; y = 64'h0000000000000002; end
            
            // Powers of 2
            5: begin x = 64'h0000000000000001; y = 64'h0000000000000001; end
            6: begin x = 64'h0000000000000002; y = 64'h0000000000000002; end
            7: begin x = 64'h0000000000000004; y = 64'h0000000000000004; end
            8: begin x = 64'h0000000000000008; y = 64'h0000000000000008; end
            9: begin x = 64'h0000000000000010; y = 64'h0000000000000010; end
            
            // Specific patterns
            10: begin x = 64'h5555555555555555; y = 64'h0000000000000002; end
            11: begin x = 64'hAAAAAAAAAAAAAAAA; y = 64'h0000000000000002; end
            
            // Random numbers
            default: begin
                x = {$random, $random};
                y = {$random, $random};
            end
        endcase

        expected = x * y;
        #10;
        if (o == expected)
            $display("Test case %d passed: %h * %h = %h", i, x, y, o);
        else
            $display("Test case %d failed. %h * %h: Expected %h, got %h", i, x, y, expected, o);
    end

    $finish;
end

// Optional: Dump waveforms
initial begin
    $dumpfile("multiplier_tb.vcd");
    $dumpvars(0, multiplier_tb);
end

endmodule