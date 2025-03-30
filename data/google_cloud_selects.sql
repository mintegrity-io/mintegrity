SELECT
    t.hash AS transaction_hash,
    t.from_address,
    t.to_address,
    t.value,
    t.gas_price,
    t.block_timestamp,
    c.address AS contract_address
FROM
    bigquery-public-data.crypto_ethereum.transactions AS t
LEFT JOIN
  bigquery-public-data.crypto_ethereum.contracts AS c
ON
    t.to_address = c.address
WHERE
    t.block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY);