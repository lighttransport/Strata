#!/bin/bash
# Batch script handed to the scheduler by jobctl. All options and settings arrive as the scheduler arguments and -x variables.
exec bash "$CTL_AGENT"
