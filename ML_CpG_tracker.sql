-- + overlap
SELECT  c.Name
       ,cast(c.Start AS int) || '-' || cast(c.End AS int)                AS cpg_cord
       ,ga.strand
       ,ga.Chr
       ,'overlapping with gene'                 AS status
       ,ga.geneName
       ,ga.geneDescription
       ,ga.Region_start || '-' || ga.Region_end AS gene_coord
FROM _450_CPG_LIST c
JOIN geneAttributes ga
ON ga.Chr = c.Chr
WHERE ( ( cast(c.Start AS int) BETWEEN ga.Region_start AND ga.Region_end ) or (cast(c.End AS int) BETWEEN ga.Region_start AND ga.Region_end ) )
AND c.Name IN ( "cg00450304", "cg00956639", "cg03949031", "cg03960390", "cg21463554", "cg23020159", "cg09977852", "cg22788109", "cg24196814", "cg05591758", "cg13754949", "cg15959756", "cg23749739", "cg18179833", "cg04878119", "cg09099257", "cg15202251", "cg26742859", "cg16141690", "cg18569885", "cg15624725", "cg06714580", "cg19850545" )
AND ga.strand = '+'
--GROUP BY  c.cpgIdRef
UNION ALL
-- - overlap
SELECT  c.Name
       ,cast(c.Start AS int) || '-' || cast(c.End AS int)                AS cpg_cord
       ,ga.strand
       ,ga.Chr
       ,'overlapping with gene'                 AS status
       ,ga.geneName
       ,ga.geneDescription
       ,ga.Region_start || '-' || ga.Region_end AS gene_coord
FROM _450_CPG_LIST c
JOIN geneAttributes ga
ON ga.Chr = c.Chr
WHERE ( ( cast(c.Start AS int) BETWEEN ga.Region_start AND ga.Region_end ) or (cast(c.End AS int) BETWEEN ga.Region_start AND ga.Region_end ) )
AND c.Name IN ( "cg00450304", "cg00956639", "cg03949031", "cg03960390", "cg21463554", "cg23020159", "cg09977852", "cg22788109", "cg24196814", "cg05591758", "cg13754949", "cg15959756", "cg23749739", "cg18179833", "cg04878119", "cg09099257", "cg15202251", "cg26742859", "cg16141690", "cg18569885", "cg15624725", "cg06714580", "cg19850545" )
AND ga.strand = '-'
--GROUP BY  c.cpgIdRef
UNION ALL
-- - promotor
SELECT  c.Name
       ,cast(c.Start AS int) || '-' || cast(c.End AS int)                AS cpg_cord
       ,ga.strand
       ,ga.Chr
       ,'in promotor region of'                 AS status
       ,ga.geneName
       ,ga.geneDescription
       ,ga.Region_start || '-' || ga.Region_end AS gene_coord
FROM _450_CPG_LIST c
JOIN geneAttributes ga
ON ga.Chr = c.Chr
WHERE ( (cast(c.End AS int) BETWEEN (ga.Region_end) AND ga.Region_end + 3000 ) )
AND c.Name IN ( "cg00450304", "cg00956639", "cg03949031", "cg03960390", "cg21463554", "cg23020159", "cg09977852", "cg22788109", "cg24196814", "cg05591758", "cg13754949", "cg15959756", "cg23749739", "cg18179833", "cg04878119", "cg09099257", "cg15202251", "cg26742859", "cg16141690", "cg18569885", "cg15624725", "cg06714580", "cg19850545" )
AND ga.strand = '-'
--GROUP BY  c.cpgIdRef
UNION ALL
-- + promotor
SELECT  c.Name
       ,cast(c.Start AS int) || '-' || cast(c.End AS int)                AS cpg_cord
       ,ga.strand
       ,ga.Chr
       ,'in promotor region of'                 AS status
       ,ga.geneName
       ,ga.geneDescription
       ,ga.Region_start || '-' || ga.Region_end AS gene_coord
FROM _450_CPG_LIST c
JOIN geneAttributes ga
ON ga.Chr = c.Chr
WHERE ( (cast(c.End AS int) BETWEEN (ga.Region_start - 5000) AND ga.Region_start -1 ) )
AND c.Name IN ( "cg00450304", "cg00956639", "cg03949031", "cg03960390", "cg21463554", "cg23020159", "cg09977852", "cg22788109", "cg24196814", "cg05591758", "cg13754949", "cg15959756", "cg23749739", "cg18179833", "cg04878119", "cg09099257", "cg15202251", "cg26742859", "cg16141690", "cg18569885", "cg15624725", "cg06714580", "cg19850545" )
AND ga.strand = '+'
--GROUP BY  c.cpgIdRef
