#include <iostream>

using namespace std;

double F2C(double F)
{
    double C = 5 * (F - 32) / 9;
    return C ;
}

int main()
{
    double F = 100;
    double C = F2C(F);
    cout << F << " Fahrenheit is " << C << " Celcius\n";
    return 0;
}