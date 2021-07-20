* ====================================================================
* Author :	Woong Choi, woongchoi@sm.ac.kr
* 		Sookmyung Women's Univ, VLSISYS Lab.
* ====================================================================
.title	'SMU, VLSISYS Lab.'
.option	post
.option	probe
.option	measdgt = 6
.option	numdgt = 6
.option	ingold
.option	modmonte = 1
.option	mtthred = 2
.option	measform = 3
.option	mcbrief = 2
.option	abstol=1e-6 reltol=1e-6
*.option ba_file = ""

* --------------------------------------------------------------------
* [ PVT MODEL ]
* --------------------------------------------------------------------
.lib	'/setup/PDK/S28LPP_2107/FullCustom/LN28LPP_HSPICE_S00-V2.0.1.2/HSPICE/LN28LPP_Hspice.lib' NN

.param	p_vdd	= 0.9
.param	p_vss	= 0
.temp	25

* --------------------------------------------------------------------
* [ Include Files]
* --------------------------------------------------------------------
.inc	'/home/woong/test/netlist'

* --------------------------------------------------------------------
* [ Vth Params ]
* --------------------------------------------------------------------
.param	dvt_pul	= '-hvec_pul*vtsig_pu*acadm'
.param	dvt_pur	= '+hvec_pur*vtsig_pu*acadm'
.param	dvt_pgl	= '-hvec_pgl*vtsig_pg*acadm'
.param	dvt_pgr	= '+hvec_pgr*vtsig_pg*acadm'
.param	dvt_pdl	= '+hvec_pdl*vtsig_pd*acadm'
.param	dvt_pdr	= '-hvec_pdr*vtsig_pd*acadm'

.param	vtsig_pu = 0.035
.param	vtsig_pg = 0.035
.param	vtsig_pd = 0.035

.param	hvec_pul =    0.137144
.param	hvec_pur =    0.254353
.param	hvec_pgl =    0.186888
.param	hvec_pgr =    0.137058
.param	hvec_pdl =    0.885725
.param	hvec_pdr =    0.388569

.param	acadm	= {SIM_ACADM}
* --------------------------------------------------------------------
* [ Circuit Description ]
* --------------------------------------------------------------------
.param	fo4_dly	= 0.01n
.param	ast_vbs = 1.0
.param	ast_wlu = 0.7
.param	ast_sbl = 1.0

vvss	vss	0	p_vss

.param	blcap =	15
cblt	blt		0	blcap*1f
cblb	blb		0	blcap*1f

.ic     v(q)	p_vss
.ic     v(qb)	p_vdd

.param	hslope	= 'fo4_dly*1'
.param	tprd	= 'fo4_dly*10'
.param	tprd0	= 'fo4_dly*4'
.param	tprd1	= 'tprd-tprd0'
.param	tplss	= 10n
.param	tplsm	= 'tplss+2*hslope+tprd0'
.param	tplse	= 'tplsm+2*hslope+tprd1'

vvdd	vdd	    0	pwl
+0n					p_vdd
+'tplss-hslope'		p_vdd
+'tplss+hslope'		'p_vdd*ast_vbs'
+'tplsm-hslope'		'p_vdd*ast_vbs'
+'tplsm+hslope'		p_vdd
+'tplse-hslope'		p_vdd
+'tplse+hslope'		p_vdd

vwl		wl	    0	pwl
+0n					p_vss
+'tplss-hslope'		p_vss
+'tplss+hslope'		'p_vdd*ast_wlu'
+'tplsm-hslope'		'p_vdd*ast_wlu'
+'tplsm+hslope'		p_vdd
+'tplse-hslope'		p_vdd
+'tplse+hslope'		p_vss

.ic		v(blt)	'p_vdd*ast_sbl'
.ic		v(blb)	'p_vdd*ast_sbl'

.param	opt_par	= 10

.model	optmod	opt	method = passfail Relin = 1e-9	itropt = 1000
.param	acadm = opt1(0,1,opt_par)
*.tran	0.01n	'tplse+5*hslope'	$sweep	optimize = opt1 result = vflip model = optmod
.tran	0.01n	'tplse+5*hslope'	sweep	optimize = opt1 result = vflip model = optmod
.probe	v(*)
.probe	i(*)

.meas		tran	vflip	when	par('v(q)-v(qb)'= '0.90*p_vdd' rise = 1
.meas		tran	p_dvtpul	param	dvt_pul
.meas		tran	p_dvtpur	param	dvt_pur
.meas		tran	p_dvtpgl	param	dvt_pgl
.meas		tran	p_dvtpgr	param	dvt_pgr
.meas		tran	p_dvtpdl	param	dvt_pdl
.meas		tran	p_dvtpdr	param	dvt_pdr

.end
