import polars as pl

def types(d: pl.DataFrame):
    numeric = [ c for c in d.columns if
                c.startswith('p_') or
                c.startswith('b_') or
                c.endswith('_lon') or
                c.endswith('_lat') ]
    
    numeric.extend(['ertrag (dt/ha)',
                    'ph_wert',
                    'rohprotein (% TS)',
                    'stärke (%)',
                    'zucker (% TS)',
                    'öl (%)'])
    
    d = d.with_columns(pl.col(numeric)
                       .str.replace_all('.','',literal=True)
                       .str.replace_all(',','.',literal=True)
                       .cast(pl.Float64,strict=False))
    
    datum = ['probenahme', 'dat_saat', 'dat_ernte', 'dat_düng']
    
    d = d.with_columns(pl.col(datum).str.to_date())

    booleans = [ c for c in d.columns if c.startswith('d_') ]
    booleans.extend(['versuchsfläche', 'öko/konv', 'bewässerung'])
    
    d = d.with_columns(pl.col(booleans)
                       .str.replace_all('.','',literal=True)
                       .str.replace_all(',','.',literal=True)
                       .str.to_integer().cast(pl.Boolean))

    return d
