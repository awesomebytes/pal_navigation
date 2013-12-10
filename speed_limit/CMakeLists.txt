cmake_minimum_required(VERSION 2.8.3)
project(speed_limit)

find_package(catkin REQUIRED)

catkin_package()

install(PROGRAMS scripts/speed_limit.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
