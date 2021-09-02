/* 
	@author: 		Rik Ghosh
	@copyright: 	Copyright 2021, University of Texas at Austin
	@credits: 		Soham Saha, Larissa Franco
	@license: 		MIT
	@version: 		1.0.5
	@maintainer: 	Rik Ghosh
	@email: 		rikghosh487@gmail.com
	@status: 		production

	@description: 	Template SQL query to use for obtaining data from the GAIA EDR3 data archive
	@link:			https://gea.esac.esa.int/archive/
	@usecase:		Bootstrap to required known or unknown cluster to generate a dataset for the pipeline
	@format:		Must produce a .CSV
*/

SELECT
	-- identification data
	gaia_source.source_id,
	-- astrometric data and errors
	gaia_source.ra,
	gaia_source.ra_error,
	gaia_source.dec,
	gaia_source.dec_error,
	gaia_source.parallax,
	gaia_source.parallax_error,
	gaia_source.pm,
	gaia_source.pmra,
	gaia_source.pmra_error,
	gaia_source.pmdec,
	gaia_source.pmdec_error,
	-- correlation data
	gaia_source.parallax_pmra_corr,
	gaia_source.parallax_pmdec_corr,
	gaia_source.pmra_pmdec_corr,
	-- photometric data
	gaia_source.phot_g_mean_mag,
	gaia_source.phot_bp_mean_mag,
	gaia_source.phot_rp_mean_mag,
	gaia_source.bp_rp,
	gaia_source.bp_g,
	gaia_source.g_rp

FROM gaiaedr3.gaia_source

WHERE
	CONTAINS(
		POINT('ICRS', gaiaedr3.gaia_source.ra, gaiaedr3.gaia_source.dec),
		CIRCLE('ICRS', <center ra>, <center dec>, <cone search radius>)		-- recommended radius: 1 to 2.5
	)=1
