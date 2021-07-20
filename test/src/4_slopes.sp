* ====================================================================
* Author :	Woong Choi,
* 		Korea Univ, VLSI Signal Processing Lab.
* ====================================================================
.title	'Korea Univ., VLSI SP Lab.'

* --------------------------------------------------------------------
* [ Options ]
* --------------------------------------------------------------------
.option	post probe
.option	modmonte = 1
.option	measdgt = 6
.option	numdgt = 6
.option	mtthred = 2
.option	measform = 3
.option	mcbrief = 2
.option	abstol=1e-6 reltol=1e-6

* --------------------------------------------------------------------
* [ Vth Params ]
* --------------------------------------------------------------------
.param	dvtpul	= '-sigval*vtsig_pu'
.param	dvtpur	= '+sigval*vtsig_pu'
.param	dvtpgl	= '-sigval*vtsig_pg'
.param	dvtpgr	= '+sigval*vtsig_pg'
.param	dvtpdl	= '+sigval*vtsig_pd'
.param	dvtpdr	= '-sigval*vtsig_pd'

.param	sigfix = 0

.param	vtsig_pu = {VTSIG_PU:0.035}
.param	vtsig_pg = {VTSIG_PG:0.035}
.param	vtsig_pd = {VTSIG_PD:0.035}

.param	hvec_pul =	{HVEC_PUL:@}
.param	hvec_pur =	{HVEC_PUR:@}
.param	hvec_pgl =	{HVEC_PGL:@}
.param	hvec_pgr =	{HVEC_PGR:@}
.param	hvec_pdl =	{HVEC_PDL:@}
.param	hvec_pdr =	{HVEC_PDR:@}

.param	mvec_pul =	{MVEC_PUL:@}
.param	mvec_pur =	{MVEC_PUR:@}
.param	mvec_pgl =	{MVEC_PGL:@}
.param	mvec_pgr =	{MVEC_PGR:@}
.param	mvec_pdl =	{MVEC_PDL:@}
.param	mvec_pdr =	{MVEC_PDR:@}

.param	nvec	 =	{NVEC:@}

* --------------------------------------------------------------------
* [ PVT MODEL ]
* --------------------------------------------------------------------
.hdl	"/home/woong/project/PTM-MG/BSIMCMG107.0.0_20130712/code/bsimcmg.va"
.lib	'/home/woong/project/PTM-MG/mpfp_sram_models' ptm14lstp_mpfp_sram
.inc	'/home/woong/project/sim/python/net/sram_6t_cell/netlist'
.param	p_vdd	= {VDD:0.30}
.param	p_vss	= 0
.temp		{TEMP:090}

* --------------------------------------------------------------------
* [ Circuit Description ]
* --------------------------------------------------------------------
.param	ast_vbs = {VBS:1.00}
.param	ast_wlu = {WLU:1.00}
.param	ast_sbl = {SBL:1.00}

vvdd	vdd	0	'p_vdd*ast_vbs'
vvss	vss	0	p_vss

.param	p_vcrit	= 0
vcrit	q	0	p_vcrit

.nodeset	v(q)	p_vss
.nodeset	v(qb)	p_vdd

vwl		wl		0	'p_vdd*ast_wlu'
vbl		bl		0	'p_vdd*ast_sbl'
vblb		blb	0	'p_vdd*ast_sbl'

* --------------------------------------------------------------------
* [ Icrit Slope Extract ]
* --------------------------------------------------------------------
.dc		p_vcrit	0	p_vdd	0.001
+sweep	sigval	0	10		0.5
.probe	v(*)
.probe	i(*)
.meas	dc	icrit		max		par('-i(vcrit)') from = 0 to = 'p_vdd/2' goal = 0

.alter	SLOPE_NATURAL
.param	dvtpul	= '-nvec*sigval*vtsig_pu'
.param	dvtpur	= '+nvec*sigval*vtsig_pu'
.param	dvtpgl	= '-nvec*sigval*vtsig_pg'
.param	dvtpgr	= '+nvec*sigval*vtsig_pg'
.param	dvtpdl	= '+nvec*sigval*vtsig_pd'
.param	dvtpdr	= '-nvec*sigval*vtsig_pd'

.alter	SLOPE_MOSTPROB
.param	dvtpul	= '-mvec_pul*sigval*vtsig_pu'
.param	dvtpur	= '+mvec_pur*sigval*vtsig_pu'
.param	dvtpgl	= '-mvec_pgl*sigval*vtsig_pg'
.param	dvtpgr	= '+mvec_pgr*sigval*vtsig_pg'
.param	dvtpdl	= '+mvec_pdl*sigval*vtsig_pd'
.param	dvtpdr	= '-mvec_pdr*sigval*vtsig_pd'

.alter	SLOPE_HYBRID
.param	dvtpul	= '-hvec_pul*sigval*vtsig_pu'
.param	dvtpur	= '+hvec_pur*sigval*vtsig_pu'
.param	dvtpgl	= '-hvec_pgl*sigval*vtsig_pg'
.param	dvtpgr	= '+hvec_pgr*sigval*vtsig_pg'
.param	dvtpdl	= '+hvec_pdl*sigval*vtsig_pd'
.param	dvtpdr	= '-hvec_pdr*sigval*vtsig_pd'

.alter	SLOPE_PUL
.param	dvtpul	= '-sigval*vtsig_pu'
.param	dvtpur	= '+sigfix*vtsig_pu'
.param	dvtpgl	= '-sigfix*vtsig_pg'
.param	dvtpgr	= '+sigfix*vtsig_pg'
.param	dvtpdl	= '+sigfix*vtsig_pd'
.param	dvtpdr	= '-sigfix*vtsig_pd'

.alter	SLOPE_PUR
.param	dvtpul	= '-sigfix*vtsig_pu'
.param	dvtpur	= '+sigval*vtsig_pu'
.param	dvtpgl	= '-sigfix*vtsig_pg'
.param	dvtpgr	= '+sigfix*vtsig_pg'
.param	dvtpdl	= '+sigfix*vtsig_pd'
.param	dvtpdr	= '-sigfix*vtsig_pd'

.alter	SLOPE_PGL
.param	dvtpul	= '-sigfix*vtsig_pu'
.param	dvtpur	= '+sigfix*vtsig_pu'
.param	dvtpgl	= '-sigval*vtsig_pg'
.param	dvtpgr	= '+sigfix*vtsig_pg'
.param	dvtpdl	= '+sigfix*vtsig_pd'
.param	dvtpdr	= '-sigfix*vtsig_pd'

.alter	SLOPE_PGR
.param	dvtpul	= '-sigfix*vtsig_pu'
.param	dvtpur	= '+sigfix*vtsig_pu'
.param	dvtpgl	= '-sigfix*vtsig_pg'
.param	dvtpgr	= '+sigval*vtsig_pg'
.param	dvtpdl	= '+sigfix*vtsig_pd'
.param	dvtpdr	= '-sigfix*vtsig_pd'

.alter	SLOPE_PDL
.param	dvtpul	= '-sigfix*vtsig_pu'
.param	dvtpur	= '+sigfix*vtsig_pu'
.param	dvtpgl	= '-sigfix*vtsig_pg'
.param	dvtpgr	= '+sigfix*vtsig_pg'
.param	dvtpdl	= '+sigval*vtsig_pd'
.param	dvtpdr	= '-sigfix*vtsig_pd'

.alter	SLOPE_PDR
.param	dvtpul	= '-sigfix*vtsig_pu'
.param	dvtpur	= '+sigfix*vtsig_pu'
.param	dvtpgl	= '-sigfix*vtsig_pg'
.param	dvtpgr	= '+sigfix*vtsig_pg'
.param	dvtpdl	= '+sigfix*vtsig_pd'
.param	dvtpdr	= '-sigval*vtsig_pd'

.end
