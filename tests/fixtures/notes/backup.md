# Backup setup

The backup runs nightly at 03:00 and is configured with borgmatic against an
external drive mounted at /mnt/archive. Retention keeps seven daily archives,
four weekly and six monthly.

Restoring a single file means listing the archive first, then extracting only
the path you need rather than the whole snapshot.
