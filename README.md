# CZ4032

## How to run data parser?
- Create a ```mysql_config.json``` in ```data_parser``` directory which includes:

```json
{
  "user": "admini9eesFs",
  "password": "ue1tA9tcZ4JZ",
  "host": "127.0.0.1",              // Use Port Forwarding
  "port": 3307,                     // Use Port Forwarding
  "db": "cz4032"
}
```

- More information on Openshift Port Forwarding can be found in https://blog.openshift.com/getting-started-with-port-forwarding-on-openshift/
- Make sure you run the ```data_cleaner``` first to generate ```users_cleaned.csv``` and ```words_cleaned.csv```.

**Note:** the value ```NA``` is somehow considered as Pandas ```NaN``` value!!
