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
.lib	'{SIM_LIB}' {SIM_PVT_P}

.param	p_vdd	= {SIM_PVT_V}
.param	p_vss	= 0
.temp	{SIM_PVT_T}

* --------------------------------------------------------------------
* [ Include Files]
* --------------------------------------------------------------------
.inc	'{SIM_NET}'

* --------------------------------------------------------------------
* [ Vth Params ]
* --------------------------------------------------------------------
.param	dvt_pul	= '-hvec_pul*vtsig_pu*acadm'
.param	dvt_pur	= '+hvec_pur*vtsig_pu*acadm'
.param	dvt_pgl	= '-hvec_pgl*vtsig_pg*acadm'
.param	dvt_pgr	= '+hvec_pgr*vtsig_pg*acadm'
.param	dvt_pdl	= '+hvec_pdl*vtsig_pd*acadm'
.param	dvt_pdr	= '-hvec_pdr*vtsig_pd*acadm'

.param	vtsig_pu = {SIM_VTSIG_PU}
.param	vtsig_pg = {SIM_VTSIG_PG}
.param	vtsig_pd = {SIM_VTSIG_PD}

.param	hvec_pul = {SIM_HVEC_PUL}
.param	hvec_pur = {SIM_HVEC_PUR}
.param	hvec_pgl = {SIM_HVEC_PGL}
.param	hvec_pgr = {SIM_HVEC_PGR}
.param	hvec_pdl = {SIM_HVEC_PDL}
.param	hvec_pdr = {SIM_HVEC_PDR}

.param	acadm	= {SIM_ACADM}
* --------------------------------------------------------------------
* [ Circuit Description ]
* --------------------------------------------------------------------
.param	fo4_dly	= {SIM_FO4DLY}
.param	ast_vbs = {SIM_AST_VBS}
.param	ast_wlu = {SIM_AST_WLU}
.param	ast_sbl = {SIM_AST_SBL}

vvss	vss	0	p_vss

.param	blcap =	{SIM_BLCAP}
cblt	blt		0	blcap*1f
cblb	blb		0	blcap*1f

.ic     v(q)	p_vss
.ic     v(qb)	p_vdd

.param	hslope	= 'fo4_dly*1'
.param	tprd	= 'fo4_dly*{SIM_AST_TOTAL}'
.param	tprd0	= 'fo4_dly*{SIM_AST_VALID}'
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

.param	opt_par	= {SIM_OPTRANGE}

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
