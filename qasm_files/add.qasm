OPENQASM 2.0;
include "qelib1.inc";
qreg q2[2];
qreg q3[2];
qreg q4[2];
h q4[0];
h q4[1];
ccx q3[0],q4[0],q3[1];
cx q4[0],q3[0];
cx q4[1],q3[1];
ccx q3[0],q2[0],q3[1];
cx q2[0],q3[0];
cx q2[1],q3[1];
