#! /usr/local/python3.8.6/bin/python3

import	os, sys, re, csv, json
import	argparse

# ----------------------------------------------------------------------
#	Colors
# ----------------------------------------------------------------------
BLACK	= '\033[30m'
RED		= '\033[31m'
GREEN	= '\033[32m'
YELLOW	= '\033[33m'
BLUE	= '\033[34m'
MAGENTA	= '\033[35m'
CYAN	= '\033[36m'
RESET	= '\033[0m'
SEL		= '\033[7m'

# ----------------------------------------------------------------------
#	Simulation Configuration (Don't touch below)
# ----------------------------------------------------------------------
srcList			= ['0_scrit.sp', '1_perturb_vec.sp', '2_acadm.sp', '3_dcadm.sp', '4_slopes.sp']
varRegExpr		= r'(?P<key>{\S+})'
#srcRegExpr		= r'{(?P<key>\S+):(?P<value>\S+)}'

# ----------------------------------------------------------------------
#	User Configuration
# ----------------------------------------------------------------------
srcPath			= './src'
simPath			= './sim'

dvtList			= ['dvt_pul', 'dvt_pur', 'dvt_pgl', 'dvt_pgr', 'dvt_pdl', 'dvt_pdr']

# Please refer to the absolute path
simLibFile		= '/setup/PDK/S28LPP_2107/FullCustom/LN28LPP_HSPICE_S00-V2.0.1.2/HSPICE/LN28LPP_Hspice.lib'
simNetFile		= '/home/woong/test/netlist'

simPvtP			= 'NN'
simPvtV			= '0.9'
simPvtT			= '25'
simVtSigPU 		= '0.035'
simVtSigPG 		= '0.035'
simVtSigPD 		= '0.035'
simAstVBS		= '1.0'
simAstWLU		= '1.0'
simAstSBL		= '1.0'
simVecRatio		= '0.8'
simFO4DLY		= '0.01n'
simAstTotal		= '10'
simAstValid		= '4'
simBlCap		= '15'
simOptRange		= '10'
# ----------------------------------------------------------------------
#	Parser	
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description='ADM/WRM simulator')

parser.add_argument('--check',			action='store_true',							help='Check the simulation is runnable')
parser.add_argument('--dvt',			type=list,				default=dvtList,		help='Delta Vth variables')
parser.add_argument('--src_list',		type=list,				default=srcList,		help='Spice Source File List')
parser.add_argument('--src_path',		type=str,				default=srcPath,		help='Spice Source File Path')
parser.add_argument('--var_regexpr',	type=str,				default=varRegExpr,		help='Regular Expression for Variables in Spice Source File')
parser.add_argument('--sim_lib',		type=str,				default=simLibFile,		help='SPICE library file')
parser.add_argument('--sim_net',		type=str,				default=simNetFile,		help='SPICE netlist file')
parser.add_argument('--sim_path',		type=str,				default=simPath,		help='Spice Run Path')
parser.add_argument('--sim_pvt_p',		type=str,				default=simPvtP,		help='Process Corner', choices=['SS', 'NN', 'FF', 'SF', 'FS'])
parser.add_argument('--sim_pvt_v',		type=str,				default=simPvtV,		help='Supply Voltage')
parser.add_argument('--sim_pvt_t',		type=str,				default=simPvtT,		help='Temperatur')
parser.add_argument('--sim_vtsig_pu',	type=str,				default=simVtSigPU,		help='One sigma value of the pull-up	PMOS VTH variation')
parser.add_argument('--sim_vtsig_pg',	type=str,				default=simVtSigPG,		help='One sigma value of the access 	NMOS VTH variation')
parser.add_argument('--sim_vtsig_pd',	type=str,				default=simVtSigPD,		help='One sigma value of the pull-down	NMOS VTH variation')
parser.add_argument('--sim_ast_vbs',	type=str,				default=simAstVBS,		help='Read Assist: VDD Boosting')
parser.add_argument('--sim_ast_wlu',	type=str,				default=simAstWLU,		help='Read Assist: WL Under Drive')
parser.add_argument('--sim_ast_sbl',	type=str,				default=simAstSBL,		help='Read Assist: Suppressed BL')
parser.add_argument('--sim_ast_total',	type=str,				default=simAstTotal,	help='SRAM Assist: Total Pulse-Width')
parser.add_argument('--sim_ast_valid',	type=str,				default=simAstValid,	help='SRAM Assist: Assist Pulse-Width')
parser.add_argument('--sim_vec_ratio',	type=str,				default=simVecRatio,	help='Most Probable Vector Ratio corresponding to the Natural Vector')
parser.add_argument('--sim_fo4dly',		type=str,				default=simFO4DLY,		help='FO4 Delay')
parser.add_argument('--sim_blcap',		type=str,				default=simBlCap,		help='BL Capacitance')
parser.add_argument('--sim_optrange',	type=str,				default=simOptRange,	help='ADM Sigma Range')

# ----------------------------------------------------------------------
#	Functions
# ----------------------------------------------------------------------
def	checkNetlist(args):
	print(CYAN, end='')
	print(f'======================================================================')
	print(f' Netlist Check')
	print(f'======================================================================')
	print(RESET, end='')
	err_flag = False
	with open(args.net, 'r') as f:
		ctx = f.read()
		for dvt_param in args.dvt:
			if dvt_param not in ctx:
				print(f' - [DVT Parameter] "{dvt_param}" is not included in the netlist file')
				err_flag = True
		for param in ['vdd', 'wl', 'q', 'qb', 'blt', 'blb']:
			if param not in ctx:
				print(f' - [Net] "{param}" is not specified in the netlist file')
				err_flag = True
	if err_flag:
		os.system(f'gvim -p {args.net}')

def	getFileList(path, end_sw=''):
	file_list = sorted( os.listdir(path) )
	return [file for file in file_list if file.endswith(end_sw)]

def	getFileVariable(infile, args):
	with open(infile, 'r') as fh:
		ctx			= fh.read()
		file_vars	= re.findall(args.var_regexpr, ctx)
	return file_vars

def	genSpiceFile(infile, outfile, simDict):
	with open(infile, 'r') as fh:
		ctx			= fh.read()
		for key in simDict.keys():
			ctx = ctx.replace('{' + key + '}', simDict[key])
	with open(outfile, 'w') as fh:
		fh.write(ctx)

def	runSpiceFile(infile, args):
	os.system(f'hspice64 -i {infile} -o {args.sim_path} > /dev/null 2>&1')

def	genDictFromMeas(infile):
	value_flag = 0
	with open(infile, 'r') as fh:
		lines = fh.read().splitlines()
	keys	= lines[-2].split(',')
	values	= lines[-1].split(',')
	return dict(zip(keys, values))

class	cssADM:
	def	__init__(self, args):
		self.srcFiles	= getFileList(args.src_path)
		self.simDict	= {}
		for arg in dir(args):
			if 'sim' in arg:
				self.simDict[arg.upper()] = eval(f'args.{arg}')
	def STEP1(self, args):
		print(RED, end='')
		print('======================================================================')
		print(' Step 1: Find SCRIT')
		print('======================================================================')
		print(RESET, end='')
		srcFile = args.src_path + '/' + self.srcFiles[0]
		runFile = args.sim_path + '/' + self.srcFiles[0]
		genSpiceFile(srcFile, runFile, self.simDict)
		runSpiceFile(runFile, args)
		step1Dict = genDictFromMeas(args.sim_path + '/0_scrit.ms0.csv')
		self.simDict['SIM_SCRIT'] = step1Dict['sigval']
		print(f' - SCRIT: {step1Dict["sigval"]}')
		return	step1Dict['sigval']

	def STEP2(self, args):
		print(RED, end='')
		print('======================================================================')
		print(' Step 2: Build Hybrid Perturbation Vector')
		print('======================================================================')
		print(RESET, end='')
		srcFile = args.src_path + '/' + self.srcFiles[1]
		runFile = args.sim_path + '/' + self.srcFiles[1]
		genSpiceFile(srcFile, runFile, self.simDict)
		runSpiceFile(runFile, args)
		step2Dict = genDictFromMeas(args.sim_path + '/1_perturb_vec.ms6.csv')
		self.simDict['SIM_HVEC_PUL'] = step2Dict['vechybrid_pul']
		self.simDict['SIM_HVEC_PUR'] = step2Dict['vechybrid_pur']
		self.simDict['SIM_HVEC_PGL'] = step2Dict['vechybrid_pgl']
		self.simDict['SIM_HVEC_PGR'] = step2Dict['vechybrid_pgr']
		self.simDict['SIM_HVEC_PDL'] = step2Dict['vechybrid_pdl']
		self.simDict['SIM_HVEC_PDR'] = step2Dict['vechybrid_pdr']
		print(f' - HVEC_PUL: {step2Dict["vechybrid_pul"]}')
		print(f' - HVEC_PUR: {step2Dict["vechybrid_pur"]}')
		print(f' - HVEC_PGL: {step2Dict["vechybrid_pgl"]}')
		print(f' - HVEC_PGR: {step2Dict["vechybrid_pgr"]}')
		print(f' - HVEC_PDL: {step2Dict["vechybrid_pdl"]}')
		print(f' - HVEC_PDR: {step2Dict["vechybrid_pdr"]}')
	
	def STEP3(self, args):
		print(RED, end='')
		print('======================================================================')
		print(' Step 3: Simulate acADM')
		print('======================================================================')
		print(RESET, end='')
		srcFile = args.src_path + '/' + self.srcFiles[2]
		runFile = args.sim_path + '/' + self.srcFiles[2]
		genSpiceFile(srcFile, runFile, self.simDict)
		runSpiceFile(runFile, args)
		step3Dict = genDictFromMeas(args.sim_path + '/2_acadm.mt0.csv')
		print(f' - acADM: {step3Dict["acadm"]}')
		return	step3Dict['acadm']


# ----------------------------------------------------------------------
#	Main
# ----------------------------------------------------------------------
def main():
	args = parser.parse_args()
	if args.check:
		checkNetlist(args)
	
	ADM = cssADM(args)
	scrit	= ADM.STEP1(args)
	ADM.STEP2(args)
	acadm	= ADM.STEP3(args)

if __name__ == '__main__':
	main()
