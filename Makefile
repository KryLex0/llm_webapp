topdir := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))/..)

run:
# main.py is located in fastapi folder
	cd fastapi && uvicorn main:app --reload

run-dev:
	cd fastapi && fastapi dev main.py