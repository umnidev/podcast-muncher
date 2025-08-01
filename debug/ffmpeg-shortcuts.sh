#!/bin/bash


# shorten audio file to 120 seconds
ffmpeg -i wee_614.m4a -t 3600 wee_614_3600.wav 
ffmpeg -i wee_614.m4a -ss 7190 -t 3600 wee_614_part3.wav 