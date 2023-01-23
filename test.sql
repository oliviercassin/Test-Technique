SELECT SUM(prod_price*prod_qty), date
    FROM TRANSACTIONS
WHERE date BETWEEN '01/01/2019' AND '31/12/2019'
GROUP BY date
ORDER BY date ASC ;

SELECT client_id, q1.ventes*prod_qty AS ventes_meuble, q2.ventes*prod_qty AS ventes_deco
    FROM (  SELECT COUNT(*) AS ventes, client_id
            FROM TRANSACTIONS NATURAL JOIN PRODUCT_NOMENCLATURE
            WHERE product_type = 'MEUBLE') q1
    JOIN (  SELECT COUNT(*) AS ventes, client_id
            FROM TRANSACTIONS NATURAL JOIN PRODUCT_NOMENCLATURE
            WHERE product_type = 'DECO') q2
        ON q1.client_id = q2.client_id
    JOIN TRANSACTIONS 
        ON TRANSACTIONS.client_id = q2.client_id
WHERE date BETWEEN '01/01/2019' AND '31/12/2019';



