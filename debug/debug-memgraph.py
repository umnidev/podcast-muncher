from neo4j import GraphDatabase
 
URI = "bolt://localhost:7687"
AUTH = ("", "")
 
memgraph = GraphDatabase.driver(URI, auth=AUTH)
memgraph.verify_connectivity()

# Drop all data in the database
# records, summary, keys = memgraph.execute_query(
#     "MATCH (n) DETACH DELETE n"
# )
# print(f"Deleted all nodes and relationships: {summary}")

# Create constraint using session with autocommit
with memgraph.session() as session:
    result = session.run("CREATE CONSTRAINT ON (p:Podcast) ASSERT p.url IS UNIQUE")
    print(f"Constraint creation result: {result.consume()}")

# Show constraints
with memgraph.session() as session:
    result = session.run("SHOW CONSTRAINT INFO")
    for record in result:
        print(f"Constraint: {record}")


# Count the number of nodes in the database
records, summary, keys = memgraph.execute_query(
    "MATCH (p:Podcast) RETURN p, count(p) AS count",
)
for record in records:
    print(record)
    print(record.data())
    print(record["p"])

records, summary, keys = memgraph.execute_query(
    "CREATE (p:Podcast {url: $url}) RETURN p",
    url="https://example.com/podcast2"
)
print(f"\nCreated Podcast node: {records} {summary} {keys}")
node = records[0]["p"]
print(f"Node props: {node._properties}")