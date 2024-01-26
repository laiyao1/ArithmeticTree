# Arithmetic Module Design

Reinforcement Learning method for designing arithmetic hardware modules, including adders and multipliers.

## Usage
- Adder design (theoretical metrics)

```
python adder.py --input_bit=64 --level_bound_delta=0
python adder.py --input_bit=128 --level_bound_delta=0
python adder.py --input_bit=128 --level_bound_delta=0 --demo
```

- Adder design (practical metrics)

(Make sure the *utils/fast_flow.tcl* and *utils/full_flow.tcl* has been copied to *OpenROAD/test*)
```
python adder.py --input_bit=32 --adder_type=0 --openroad_path=/path/to/openroad
python adder.py --input_bit=32 --adder_type=1 --openroad_path=/path/to/openroad
python adder.py --input_bit=32 --adder_type=2 --openroad_path=/path/to/openroad
python select_adder.py --input_bit=32 --openroad_path=/path/to/openroad
```

- Multiplier design
```
python mult.py --input_bit=32 --area_w=0.01
python mult.py --input_bit=64 --area_w=0.01
python mult.py --input_bit=128 --area_w=0.01
```


## Expected output

- For adder design (theorical metrics), the designed adders are represented by matrix $A_{N×N}$ and are dumped into the dir *cell_map/adder_{N}b/adder_{N}b_{L}l_{S}s_{ID}.log*, where $N$ is the bit number of adders, $L$ is the level, $S$ is the size and $ID$ is the id for adders. At the same time, timing reports are saved in *time_log_{N}b_{L}_{SEED}.log*.
- For adder design (practical metrics), the designed adders are saved as verilog HDL files (\*.v) and cell map matrix (\*.log). Because the verilog HDL files , it only saves cell map files by default. If you want to save Verilog HDL files, switch on the '--save_verilog'. At the same time, logs are written in *adder_prac_log/adder_{N}b/adder_{N}b_openroad_type{TID}_{TIME}.log*, where *{TID}* is the type of the initial adder as follows and *{TIME}* is the time when the program starts running.

|Adder Type ID | Adder|
|---|---|
|0|ripple carry adder|
|1|Sklansky Adder|
|2|Brent-Kung Adder|

- For multiplier design, the best-designed multipliers are logged in the dir *back_and_forth*. At the same time, the Verilog HDL files are in dir *run_verilog_mult_mid* and *run_verilog_mult_add_mid*.
- For optimizing theoretical metrics in adder design, advanced results can be found in less than 3 hours in our experimental hardware.
- For other design tasks, the estimated running time is listed in Table A3.


## Demo Output
1. Adder design (theoretical metris):

```
python adder.py --input_bit=128 --level_bound_delta=0
```

For adder outputs:
<!-- - *cell_map/adder_128b/adder_128b_8l_364s_0.log* -->
```
1 0 0 0 0 0 ...
1 1 0 0 0 0 ...
1 0 1 0 0 0 ...
1 0 1 1 0 0 ...
1 0 0 0 1 0 ...
1 0 0 0 1 1 ...
...
1 0 0 0 0 0 ...
```


For timing log outputs:
<!-- - *time_log_128b_0_1.log* -->

- Output format:

|Level| Size| Clock Time | Search Step|
| ---| ---| ---| ---|
```
...
8.0 369.0   1.36    83
8.0 368.0   6.94    332
8.0 367.0   7.20    333
8.0 366.0   87.06   3810
8.0 365.0   148.33  6353
8.0 364.0   1669.44 71038
```
Columns are level, size, clock time and search steps. 

Actual run results are subject to hardware and random seed variations.
This is another timing reports.
```
...
8.0 369.0   31.90   565
8.0 368.0   40.90   726
8.0 367.0   41.41   727
8.0 366.0   997.32  16771
8.0 365.0   1774.52 29994
8.0 364.0   6128.24 105726
``` 

When level_bound_delta=1, the timing reports:
```
...
9.0 279.0   1.96    91
9.0 278.0   1.98    92
9.0 277.0   23.35   1008
9.0 276.0   23.51   1010
9.0 275.0   23.68   1011
9.0 274.0   1119.71 61278
9.0 273.0   1119.88 61279
```

(Because files are large, '...' Indicates omitted file contents)

1. Adder design (theoretical metris):

```
python adder.py --input_bit=128 --level_bound_delta=0
```

For verilog HDL outputs:
```
module main(a,b,s,cout);
input [31:0] a,b;
output [31:0] s;
output cout;
wire g11_8,g26_16,p8_8,c4,...
assign p0_0 = a[0] ^ b[0];
assign g0_0 = a[0] & b[0];
...
```

For log outputs:

- Output format:

|File Name | Delay($ps$) | Area($um^2$) | * | Level | Size| Fanout| Step| Cache Hit| *| Timing ($s$)|
| ---| --- | ---| ---|---|---|---|---|---|---|---|

\* are fields reserved for future use.


```
adder_32b_7_79_70199b11a9eeffb33f33691274f08116 450.97  401.13  0.0 7   79  16  1   0   0   1.23
adder_32b_8_82_8b6eaa7a64bf15b50f9b8abbb51d7145 408.76  416.82  0.0 8   82  15  2   0   0   1.86
adder_32b_8_84_583cb9fc4b850f0be7be0df992a414c2 411.60  441.83  0.0 8   84  15  3   0   0   2.50
adder_32b_8_83_b1077cbc7d258fcdb8cd10279b1f7630 412.09  423.21  0.0 8   83  15  4   0   0   3.12
adder_32b_8_84_5053d501e2c45c99f6854e08ed3a4265 405.76  430.12  0.0 8   84  14  5   0   0   3.75
```

When running *select_adder.py*, the selected top K adders are outputed into *adder_{N}b_full_openroad_{TIME}.log*, with four more columns:

|Delay($ps$, full flow) | Area($um^2$, full flow) | * | Timing ($s$)|
| ---| --- | ---| ---|



## Parameters

- **input_bit** Adder/multiplier bits.
- **initial_adder_type** Initial adder for MCTS.
- **seed** Random seed.
- **openroad_path** OpenROAD installation path.

For adder design:
- **level_bound_delta** Upper bound of level. ($+log_2N+1$)
- **max_save** Maximum number of adders stored per level threshold $L_{th}$. (Default: $2$)
- **demo** Easier access to design results from papers.
- **save_verilog** Save adder Verilog HDL files.
- **step** Maximum search steps.
- **adder_type** Type of adder for initialization.
- **k** Top K adders are selected in the 2nd stage of retrieval.


For multiplier design:
- **area_w** Weight for area.
- **max_iter** Maximum number of search steps.
- **template** Compressor tree file.
- **gamma** Decay factor.
- **lr** Learning rate.
- **batch_size** Batch size.



## Requirements
- [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) >= 2.0
- [yosys](https://github.com/YosysHQ/yosys) >= 0.27
- [Python](https://www.python.org/) >= 3.9
- [Pytorch](https://pytorch.org/) >= 1.10
- [gym](https://www.gymlibrary.dev/index.html) >= 0.21.0
- [tqdm](https://tqdm.github.io/)
- Ubuntu >= 20.04
- Other versions may also work, but not tested.

## Installation
Our codes do not need installation.
However, it is necessary to install OpenROAD and Yosys when testing multipliers (installation time: ~1 hour).

Then copy the following files for synthesis:
```
cp utils/fast_flow.tcl /path/to/OpenROAD/test
cp utils/full_flow.tcl /path/to/OpenROAD/test
```

