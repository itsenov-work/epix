-- ============================================================
-- Name: Diabetes Reports
-- Version: 1.0
-- Last edit: 12.12.2022
-- Author:     Momchil Topalov <momchil.topalov@gmail.com>
-- Source data: local copy of Disgenet + CpG annotation info
-- ============================================================

-- select all diabetes disease numenclatures present in Disgenet 
-- that belong to the '   Nutritional and Metabolic Diseases' class
-- the list will be used later on 
SELECT  da.diseaseName
FROM diseaseAttributes da
JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID
JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID
WHERE da.diseaseName like '%diabetes%'
AND dc.diseaseClassName = '   Nutritional and Metabolic Diseases'
GROUP BY  da.diseaseNID;

-- ============================================================
-- GENE QUERIES
-- clumns:
-- geneName	geneDescription	DPI	DSI	associationType	score	Chr	strand	Region_start	Region_end	ensembl
-- ============================================================

-- select all GESTATIONAL diabetes related genes
SELECT  ga.geneName
       ,ga.geneDescription
       ,ga.DPI
       ,ga.DSI
       ,gdn.associationType
       ,gdn.score
       ,ga.Chr
       ,ga.strand
       ,ga.Region_start
       ,ga.Region_end
       ,ga.ensembl
FROM geneAttributes ga
JOIN geneDiseaseNetwork gdn ON gdn.geneNID = ga.geneNID
JOIN diseaseAttributes da ON da.diseaseNID = gdn.diseaseNID
WHERE da.diseaseId IN ( 
  SELECT da.diseaseId 
  FROM diseaseAttributes da 
  JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID 
  JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID 
  WHERE da.diseaseName like '%diabetes%' 
  AND da.diseaseName like '%GESTATIONAL%' 
  AND dc.diseaseClassName = '   Nutritional and Metabolic Diseases' 
  GROUP BY da.diseaseNID )
GROUP BY  ga.geneName;

-- select all TYPE ONE diabetes related genes
SELECT  ga.geneName
       ,ga.geneDescription
       ,ga.DPI
       ,ga.DSI
       ,gdn.associationType
       ,gdn.score
       ,ga.Chr
       ,ga.strand
       ,ga.Region_start
       ,ga.Region_end
       ,ga.ensembl
FROM geneAttributes ga
JOIN geneDiseaseNetwork gdn ON gdn.geneNID = ga.geneNID
JOIN diseaseAttributes da ON da.diseaseNID = gdn.diseaseNID
WHERE da.diseaseId IN ( 
  SELECT da.diseaseId 
  FROM diseaseAttributes da 
  JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID 
  JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID 
  WHERE da.diseaseName like '%diabetes%' 
  AND (da.diseaseName like '%type 1%' or da.diseaseName like '%type I%') 
  AND dc.diseaseClassName = '   Nutritional and Metabolic Diseases' GROUP BY da.diseaseNID )
GROUP BY  ga.geneName;

-- select all TYPE TWO diabetes related genes
SELECT  ga.geneName
       ,ga.geneDescription
       ,ga.DPI
       ,ga.DSI
       ,gdn.associationType
       ,gdn.score
       ,ga.Chr
       ,ga.strand
       ,ga.Region_start
       ,ga.Region_end
       ,ga.ensembl
FROM geneAttributes ga
JOIN geneDiseaseNetwork gdn ON gdn.geneNID = ga.geneNID
JOIN diseaseAttributes da ON da.diseaseNID = gdn.diseaseNID
WHERE da.diseaseId IN ( 
  SELECT da.diseaseId 
  FROM diseaseAttributes da 
  JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID 
  JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID 
  WHERE da.diseaseName like '%diabetes%' 
  AND (da.diseaseName like '%type 2%' or da.diseaseName like '%type II%') 
  AND dc.diseaseClassName = '   Nutritional and Metabolic Diseases' GROUP BY da.diseaseNID )
GROUP BY  ga.geneName;

-- select related genes per disease from the list
SELECT  ga.geneName
       ,ga.geneDescription
       ,ga.DPI
       ,ga.DSI
       ,gdn.associationType
       ,gdn.score
       ,ga.Chr
       ,ga.strand
       ,ga.Region_start
       ,ga.Region_end
       ,ga.ensembl
FROM geneAttributes ga
JOIN geneDiseaseNetwork gdn ON gdn.geneNID = ga.geneNID
JOIN diseaseAttributes da ON da.diseaseNID = gdn.diseaseNID
WHERE da.diseaseId = (
  SELECT  daa.diseaseId
  FROM diseaseAttributes daa
  WHERE daa.diseaseName = ? ) 
  AND gdn.score > 0.1 
  AND ga.DSI > 0.5 
  AND gdn.EI is not 0
GROUP BY  ga.geneName;

-- select ALL diabetes related genes
-- the additional filters ensure specifity of the result
SELECT  ga.geneName
       ,ga.geneDescription
       ,ga.DPI
       ,ga.DSI
       ,gdn.associationType
       ,gdn.score
       ,ga.Chr
       ,ga.strand
       ,ga.Region_start
       ,ga.Region_end
       ,ga.ensembl
FROM geneAttributes ga
JOIN geneDiseaseNetwork gdn ON gdn.geneNID = ga.geneNID
JOIN diseaseAttributes da ON da.diseaseNID = gdn.diseaseNID
WHERE da.diseaseId IN ( 
  SELECT da.diseaseId 
  FROM diseaseAttributes da 
  JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID 
  JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID 
  WHERE da.diseaseName like '%diabetes%' 
  AND dc.diseaseClassName = '   Nutritional and Metabolic Diseases' 
  GROUP BY da.diseaseNID )
AND gdn.score > 0.1 -- Gene Disease Assosiation score
AND ga.DSI > 0.5 -- Disease Specificity Index
AND gdn.EI is not 0 -- Evidence Level
GROUP BY  ga.geneName;

-- ============================================================
-- VARIANT QUERIES
-- columns:
-- variantId	s	chromosome	coord	most_severe_consequence	DSI	DPI	associationType	score	EI	pmid
-- ============================================================

-- select related variants per disease from the list
SELECT  ga.geneName
       ,ga.geneDescription
       ,ga.DPI
       ,ga.DSI
       ,gdn.associationType
       ,gdn.score
       ,ga.Chr
       ,ga.strand
       ,ga.Region_start
       ,ga.Region_end
       ,ga.ensembl
FROM geneAttributes ga
JOIN geneDiseaseNetwork gdn ON gdn.geneNID = ga.geneNID
JOIN diseaseAttributes da ON da.diseaseNID = gdn.diseaseNID
WHERE da.diseaseId = (
  SELECT  daa.diseaseId
  FROM diseaseAttributes daa
  WHERE daa.diseaseName = ? ) 
  AND gdn.score > 0.1 
  AND ga.DSI > 0.5 
  AND gdn.EI is not 0
GROUP BY  ga.geneName; 

-- select all GESTATIONAL diabetes related variants
SELECT  va.variantId
       ,va.s
       ,va.chromosome
       ,va.coord
       ,va.most_severe_consequence
       ,va.DSI
       ,va.DPI
       ,vdn.associationType
       ,vdn.score
       ,vdn.EI
       ,vdn.pmid
FROM variantAttributes va
JOIN variantDiseaseNetwork vdn ON vdn.variantNID = va.variantNID
JOIN diseaseAttributes da ON da.diseaseNID = vdn.diseaseNID
WHERE da.diseaseId IN ( 
  SELECT da.diseaseId 
  FROM diseaseAttributes da 
  JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID 
  JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID 
  WHERE da.diseaseName like '%diabetes%' 
  AND da.diseaseName like '%GESTATIONAL%' 
  AND dc.diseaseClassName = '   Nutritional and Metabolic Diseases' 
  GROUP BY da.diseaseNID )
GROUP BY  va.variantNID

-- select all TYPE ONE diabetes related variants
SELECT  va.variantId
       ,va.s
       ,va.chromosome
       ,va.coord
       ,va.most_severe_consequence
       ,va.DSI
       ,va.DPI
       ,vdn.associationType
       ,vdn.score
       ,vdn.EI
       ,vdn.pmid
FROM variantAttributes va
JOIN variantDiseaseNetwork vdn ON vdn.variantNID = va.variantNID
JOIN diseaseAttributes da ON da.diseaseNID = vdn.diseaseNID
WHERE da.diseaseId IN ( 
  SELECT da.diseaseId 
  FROM diseaseAttributes da 
  JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID 
  JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID 
  WHERE da.diseaseName like '%diabetes%' 
  AND (da.diseaseName like '%type 1%' or da.diseaseName like '%type I%') 
  GROUP BY da.diseaseNID )
GROUP BY  va.variantNID

-- select all TYPE TWO diabetes related genes
SELECT  va.variantId
       ,va.s
       ,va.chromosome
       ,va.coord
       ,va.most_severe_consequence
       ,va.DSI
       ,va.DPI
       ,vdn.associationType
       ,vdn.score
       ,vdn.EI
       ,vdn.pmid
FROM variantAttributes va
JOIN variantDiseaseNetwork vdn ON vdn.variantNID = va.variantNID
JOIN diseaseAttributes da ON da.diseaseNID = vdn.diseaseNID
WHERE da.diseaseId IN ( 
  SELECT da.diseaseId 
  FROM diseaseAttributes da 
  JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID 
  JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID 
  WHERE da.diseaseName like '%diabetes%' 
  AND (da.diseaseName like '%type 2%' or da.diseaseName like '%type II%') 
  AND dc.diseaseClassName = '   Nutritional and Metabolic Diseases' 
  GROUP BY da.diseaseNID )
GROUP BY  va.variantNID

-- select ALL diabetes related variants
-- the additional filters ensure specifity of the result
SELECT  va.variantId
       ,va.s
       ,va.chromosome
       ,va.coord
       ,va.most_severe_consequence
       ,va.DSI
       ,va.DPI
       ,vdn.associationType
       ,vdn.score
       ,vdn.EI
       ,vdn.pmid
FROM variantAttributes va
JOIN variantDiseaseNetwork vdn ON vdn.variantNID = va.variantNID
JOIN diseaseAttributes da ON da.diseaseNID = vdn.diseaseNID
WHERE da.diseaseId IN ( 
  SELECT da.diseaseId FROM diseaseAttributes da 
  JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID 
  JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID 
  WHERE da.diseaseName like '%diabetes%' 
  AND dc.diseaseClassName = '   Nutritional and Metabolic Diseases' 
  GROUP BY da.diseaseNID )
AND gdn.score > 0.1 -- Gene Disease Assosiation score
AND ga.DSI > 0.5 -- Disease Specificity Index
AND gdn.EI is not 0 -- Evidence Level
GROUP BY  va.variantNID

-- ============================================================
-- CPG QUERIES
-- columns:
-- geneName	Name	AddressA_ID	Strand	Chr	Start	End
-- ============================================================

-- select all diabetes related CpG in promotor range of -1000bp +500bp
-- this 1500bp region covers most if not all CpG overlaping with 
-- the promotor region and regulate the transcription
-- the query is 2in1 with UNION ALL as it covers + and - DNA strand
SELECT  ga.geneName
       ,c.Name
       ,c.AddressA_ID
       ,c.Strand
       ,c.Start
       ,c.End
FROM _450_CPG_LIST c
JOIN geneAttributes ga ON ga.Chr = c.Chr
WHERE ( 
  ( (ga.Region_start - 1000) BETWEEN cast(c.Start AS int) AND cast(c.End AS int) ) 
  or 
  ( cast(c.Start AS int) BETWEEN (ga.Region_start - 1000) AND (ga.Region_start + 500) ) 
)
AND ga.Chr = c.Chr
AND ga.geneName IN (
  SELECT  ga.geneName
  FROM geneAttributes ga
  JOIN geneDiseaseNetwork gdn ON gdn.geneNID = ga.geneNID
  JOIN diseaseAttributes da ON da.diseaseNID = gdn.diseaseNID
  WHERE da.diseaseId IN (
    SELECT  da.diseaseId
    FROM diseaseAttributes da
    JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID
    JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID
    WHERE da.diseaseName like '%diabetes%'
    AND dc.diseaseClassName = '   Nutritional and Metabolic Diseases'
    GROUP BY  da.diseaseNID 
    ) 
    AND gdn.score > 0.1 
    AND ga.DSI > 0.5
    AND gdn.EI is not 0 
    AND ga.strand = '+'
  GROUP BY  ga.geneName 
)
UNION ALL
SELECT  ga.geneName
       ,c.Name
       ,c.AddressA_ID
       ,c.Strand
       ,c.Start
       ,c.End
FROM _450_CPG_LIST c
JOIN geneAttributes ga ON ga.Chr = c.Chr
WHERE ( 
  ( (ga.Region_end - 1000) BETWEEN cast(c.Start AS int) AND cast(c.End AS int) ) 
  or 
  ( cast(c.Start AS int) BETWEEN (ga.Region_end - 1000) AND (ga.Region_end + 500) ) 
)
AND ga.Chr = c.Chr
AND ga.geneName IN (
  SELECT  ga.geneName
  FROM geneAttributes ga
  JOIN geneDiseaseNetwork gdn ON gdn.geneNID = ga.geneNID
  JOIN diseaseAttributes da ON da.diseaseNID = gdn.diseaseNID
  WHERE da.diseaseId IN (
    SELECT  da.diseaseId
    FROM diseaseAttributes da
    JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID
    JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID
    WHERE da.diseaseName like '%diabetes%'
    AND dc.diseaseClassName = '   Nutritional and Metabolic Diseases'
    GROUP BY  da.diseaseNID 
    )
    AND gdn.score > 0.1 
    AND ga.DSI > 0.5 
    AND gdn.EI is not 0 
    AND ga.strand = '-'
  GROUP BY  ga.geneName
)

-- select all diabetes related CpG overlaping with the gene
-- covering both UTRs, all exons and introns
SELECT  ga.geneName
       ,c.Name
       ,c.AddressA_ID
       ,c.Strand
       ,c.Start
       ,c.End
FROM _450_CPG_LIST c
JOIN geneAttributes ga ON ga.Chr = c.Chr
WHERE( 
  (ga.Region_start BETWEEN c.Start AND c.End) 
  OR 
  (c.Start BETWEEN ga.Region_start AND ga.Region_end) 
)
AND ga.geneName IN (
  SELECT  ga.geneName
  FROM geneAttributes ga
  JOIN geneDiseaseNetwork gdn ON gdn.geneNID = ga.geneNID
  JOIN diseaseAttributes da ON da.diseaseNID = gdn.diseaseNID
  WHERE da.diseaseId IN (
    SELECT  da.diseaseId
    FROM diseaseAttributes da
    JOIN disease2class dtc ON dtc.diseaseNID = da.diseaseNID
    JOIN diseaseClass dc ON dc.diseaseClassNID = dtc.diseaseClassNID
    WHERE da.diseaseName like '%diabetes%'
    AND dc.diseaseClassName = '   Nutritional and Metabolic Diseases'
    GROUP BY  da.diseaseNID 
    ) 
    AND gdn.score > 0.1 
    AND ga.DSI > 0.5
    AND gdn.EI is not 0 

  GROUP BY  ga.geneName 
)