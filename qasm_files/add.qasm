OPENQASM 2.0;
include "qelib1.inc";
qreg state[2];
qreg add[2];
qreg input[2];
h input[0];
h input[1];
ccx add[0],input[0],add[1];
cx input[0],add[0];
cx input[1],add[1];
ccx add[0],state[0],add[1];
cx state[0],add[0];
cx state[1],add[1];