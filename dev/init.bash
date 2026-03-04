#!/bin/bash
git submodule init
cd freesurfer
git remote add datasrc https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/repo/annex.git 2>/dev/null || true
git fetch datasrc
git annex get mri_synthseg mri_synthstrip
