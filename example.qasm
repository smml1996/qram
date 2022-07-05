OPENQASM 2.0;
include "qelib1.inc";

qreg reg[4];
qreg scratch[1];

h reg[0];
h reg[1];
h reg[2];
h reg[3];

barrier reg[0],reg[1],reg[2],reg[3],scratch[0];
x reg[2];
x reg[3];
ccx reg[0],reg[1],scratch[0];
h reg[3];
ccx scratch[0],reg[2],reg[3];
h reg[3];
ccx reg[0],reg[1],scratch[0];
x reg[2];
x reg[3];

barrier reg[0],reg[1],reg[2],reg[3],scratch[0];
h reg[0];
h reg[1];
h reg[2];
h reg[3];
x reg[0];
x reg[1];
x reg[2];
x reg[3];
ccx reg[0],reg[1],scratch[0];
h reg[3];
ccx scratch[0],reg[2],reg[3];
h reg[3];
ccx reg[0],reg[1],scratch[0];
x reg[0];
x reg[1];
x reg[2];
x reg[3];
h reg[0];
h reg[1];
h reg[2];
h reg[3];

barrier reg[0],reg[1],reg[2],reg[3],scratch[0];
x reg[2];
x reg[3];
ccx reg[0],reg[1],scratch[0];
h reg[3];
ccx scratch[0],reg[2],reg[3];
h reg[3];
ccx reg[0],reg[1],scratch[0];
x reg[2];
x reg[3];

barrier reg[0],reg[1],reg[2],reg[3],scratch[0];
h reg[0];
h reg[1];
h reg[2];
h reg[3];
x reg[0];
x reg[1];
x reg[2];
x reg[3];
ccx reg[0],reg[1],scratch[0];
h reg[3];
ccx scratch[0],reg[2],reg[3];
h reg[3];
ccx reg[0],reg[1],scratch[0];
x reg[0];
x reg[1];
x reg[2];
x reg[3];
h reg[0];
h reg[1];
h reg[2];
h reg[3];

barrier reg[0],reg[1],reg[2],reg[3],scratch[0];
x reg[2];
x reg[3];
ccx reg[0],reg[1],scratch[0];
h reg[3];
ccx scratch[0],reg[2],reg[3];
h reg[3];
ccx reg[0],reg[1],scratch[0];
x reg[2];
x reg[3];

barrier reg[0],reg[1],reg[2],reg[3],scratch[0];
h reg[0];
h reg[1];
h reg[2];
h reg[3];
x reg[0];
x reg[1];
x reg[2];
x reg[3];
ccx reg[0],reg[1],scratch[0];
h reg[3];
ccx scratch[0],reg[2],reg[3];
h reg[3];
ccx reg[0],reg[1],scratch[0];
x reg[0];
x reg[1];
x reg[2];
x reg[3];
h reg[0];
h reg[1];
h reg[2];
h reg[3];

barrier reg[0],reg[1],reg[2],reg[3],scratch[0];
x reg[2];
x reg[3];
ccx reg[0],reg[1],scratch[0];
h reg[3];
ccx scratch[0],reg[2],reg[3];
h reg[3];
ccx reg[0],reg[1],scratch[0];
x reg[2];
x reg[3];

barrier reg[0],reg[1],reg[2],reg[3],scratch[0];
h reg[0];
h reg[1];
h reg[2];
h reg[3];
x reg[0];
x reg[1];
x reg[2];
x reg[3];
ccx reg[0],reg[1],scratch[0];
h reg[3];
ccx scratch[0],reg[2],reg[3];
h reg[3];
ccx reg[0],reg[1],scratch[0];
x reg[0];
x reg[1];
x reg[2];
x reg[3];
h reg[0];
h reg[1];
h reg[2];
h reg[3];