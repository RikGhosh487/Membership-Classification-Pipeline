# Queries
Contains ADQL and SQL queries to be used in the GAIA database for data extraction. The [Template Query](https://github.com/RikGhosh487/Membership-Classification-Pipeline.git/blob/main/queries/template.sql) contains the basis for developing ADQL queries for the GAIA database. Modifying unspecified items within `<>` the template will produce an applicable query for the required region in space that is detectable by the GAIA EDR3 database.

## Fields to Populate
- >`<center ra>`: the Right Ascension (in degrees) for the cone search
- >`<center dec>`: the Declination (in degrees) for the cone search
- >`<cone search radius>`: the radius for the cone search. We recommend using values between $1$ degrees and $2.5$ degrees

### Express Links and Tutorials
- [GAIA EDR3 Database Archive](https://gea.esac.esa.int/)
- [ADQL Tutorial](https://gea.esac.esa.int/archive-help/adql/examples/index.html)
- [SQL Tutorial](https://www.w3schools.com/sql/)
