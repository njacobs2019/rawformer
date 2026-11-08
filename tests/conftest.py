import os

# Must run before any test imports rawformer so the jaxtyping import hook is
# installed. Override with BEARTYPE=0 to run without runtime shape checks.
os.environ.setdefault("BEARTYPE", "1")
