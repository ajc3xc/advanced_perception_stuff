#!/usr/bin/env python3
"""
3b_train_roboflow.py — Kick off training on Roboflow for an existing version.
"""

from roboflow import Roboflow

rf      = Roboflow(api_key="e20yDTNrBI9q2Sst9wA3")
project = rf.workspace("xiaoyus-workspace-wtaot").project("bamboo_synthetic")
version = project.version(1)

print("Starting training …")
version.train()
print("✅ Training started — check app.roboflow.com for progress")