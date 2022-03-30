#! /bin/bash

rm -rf /var/www/html/onag/*
cp -r _site/* /var/www/html/onag/
chown -R  www-data:banserver /var/www/html/onag/


