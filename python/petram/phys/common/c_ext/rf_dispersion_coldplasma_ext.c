#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
/* The functions below are the version-1 native ABI.  They never retain a
* Python/NumPy pointer and write a row-major 3 by 3 complex128 matrix. */
#define Q0 1.60217662e-19
#define EPS0 8.8541878176e-12
#define ME 9.1093837100818e-31
#define OK 0
#define EINVAL_DIM 1
static void
put(double *output, int row, int column, double complex value)
{
    output[2 * (row * 3 + column)] = creal(value);
    output[2 * (row * 3 + column) + 1] = cimag(value);
}
static double complex
get(const double *output, int row, int column)
{
    return output[2 * (row * 3 + column)] +
    I * output[2 * (row * 3 + column) + 1];
}
static void
spd_el(double w, double b, double dens, double nu, int model,
double complex *s, double complex *p, double complex *d)
{
    if (model <= 2) {
        double complex mass = (1.0 + I * nu / w) * ME;
        double complex wp2 = dens * Q0 * Q0 / (mass * EPS0);
        double complex wc = -Q0 * b / mass;
        *p = -wp2 / (w * w);
        *s = -wp2 / (w * w - wc * wc);
        *d = wc * wp2 / (w * (w * w - wc * wc));
    }
    else {
        double complex w2 = w + I * nu;
        double complex wp2 = dens * Q0 * Q0 / (ME * EPS0);
        double wc = -Q0 * b / ME;
        double complex right = -wp2 / w / (w2 + wc);
        double complex left = -wp2 / w / (w2 - wc);
        *p = -wp2 / w / w2;
        *s = (right + left) / 2.0;
        *d = (right - left) / 2.0;
    }
}
static void spd_ion(double w, double b, double dens, double mass, int32_t z, double nu, int model, double complex *s, double complex *p, double complex *d) {
    double q=z*Q0;
    if (model <= 2) {
        double complex m=(1.0+I*nu/w)*mass, wp=dens*q*q/(m*EPS0), wc=q*b/m;
        *p=-wp/(w*w);
        *s=-wp/(w*w-wc*wc);
        *d=wc*wp/(w*(w*w-wc*wc));
    }
    else {
        double complex w2=w+I*nu, wp=dens*q*q/(mass*EPS0);
        double wc=q*b/mass, r=-wp/w/(w2+wc), l=-wp/w/(w2-wc);
        *p=-wp/w/w2;
        *s=(r+l)/2.;
        *d=(r-l)/2.;
    }
}
static void collisions(const double *dens, const double *masses, const int32_t *z, int n, double te, double ne, double *nu) {
    int k;
    for(k=0;k<=n;k++) nu[k]=0.;
    if(ne==0.) return;
    double vt=sqrt(2.*te*Q0/ME), lambda=1.+12.*M_PI*pow(EPS0*te*Q0,1.5)/(pow(Q0,3.)*sqrt(ne));
    for(k=0;k<n;k++) {
        double q=z[k]*Q0;
        nu[k+1]=q*q*Q0*Q0*dens[k]*log(lambda)/(4.*M_PI*EPS0*EPS0*ME*ME*pow(vt,3.));
        nu[0]+=nu[k+1];
    }
    for(k=0;k<n;k++) {
        double q=z[k]*Q0;
        nu[k+1]=q*q*Q0*Q0*dens[k]*log(lambda)/(4.*M_PI*EPS0*EPS0*ME*ME*pow(vt,3.))*ME/masses[k];
    }
}
static void add_spd(double *o, double complex s, double complex p, double complex d) {
    put(o,0,0,get(o,0,0)+s);
    put(o,1,1,get(o,1,1)+s);
    put(o,2,2,get(o,2,2)+p);
    put(o,0,1,get(o,0,1)-I*d);
    put(o,1,0,get(o,1,0)+I*d);
}
int petram_cold_std_v1(double w, const double *b, const double *dens, const double *masses, const int32_t *z, const double *temp, int32_t n, double ne, int32_t model, double *out) {
    if(n<0) return EINVAL_DIM;
    double bn=sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]), nu[n+1];
    double complex s,p,d;
    int k;
    if(model==1 || model==2) collisions(dens,masses,z,n,temp[0],ne,nu);
    else for(k=0;k<=n;k++) nu[k]=(model==3||model==4)?temp[k]:0.;
    if(model==1) {
        double mx=nu[0];
        for(k=1;k<=n;k++) if(nu[k]>mx) mx=nu[k];
        for(k=0;k<=n;k++) nu[k]=mx;
    }
    for(k=0;k<18;k++) out[k]=0.;
    put(out,0,0,1.);
    put(out,1,1,1.);
    put(out,2,2,1.);
    if(ne>0.) {
        spd_el(w,bn,ne,nu[0],model,&s,&p,&d);
        add_spd(out,s,p,d);
    }
    for(k=0;k<n;k++) if(dens[k]>0.) {
        spd_ion(w,bn,dens[k],masses[k],z[k],nu[k+1],model,&s,&p,&d);
        add_spd(out,s,p,d);
    }
    return OK;
}
int petram_cold_g_v1(double w, const double *b, const double *dens, const double *masses, const int32_t *z, const double *temp, int32_t n, double ne, const int32_t *terms, int32_t rows, int32_t eye, int32_t model, double *out) {
    int rc=petram_cold_std_v1(w,b,dens,masses,z,temp,n,ne,model,out), k;
    if(rc||rows<n+1) return rc?rc:EINVAL_DIM;
    /* Recompute species-wise: generalized term masks cannot be applied to summed SPD. */
    double bn=sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]), nu[n+1];
    double complex s,p,d;
    if(model==1||model==2) collisions(dens,masses,z,n,temp[0],ne,nu);
    else for(k=0;k<=n;k++)nu[k]=(model>=3)?temp[k]:0.;
    if(model==1) {
        double mx=nu[0];
        for(k=1;k<=n;k++)if(nu[k]>mx)mx=nu[k];
        for(k=0;k<=n;k++)nu[k]=mx;
    }
    for(k=0;k<18;k++)out[k]=0.;
    if(eye) {
        put(out,0,0,1);
        put(out,1,1,1);
        put(out,2,2,1);
    }
    for(k=0;k<=n;k++) {
        if(k==0) {
            if(ne<=0)continue;
            spd_el(w,bn,ne,nu[0],model,&s,&p,&d);
        }
        else {
            if(dens[k-1]<=0)continue;
            spd_ion(w,bn,dens[k-1],masses[k-1],z[k-1],nu[k],model,&s,&p,&d);
        }
        const int32_t *t=terms+5*k;
        if(!t[0])s=0.;
        if(!t[1])d=0.;
        if(!t[2])p=0.;
        if(!t[3]) {
            s=I*cimag(s);
            p=I*cimag(p);
            d=I*cimag(d);
        }
        if(!t[4]) {
            s=creal(s);
            p=creal(p);
            d=creal(d);
        }
        add_spd(out,s,p,d);
    }
    return OK;
}
static PyObject *cold_std_py(PyObject *self, PyObject *args) {
    double w,ne;
    int model;
    PyObject *bo,*do_,*mo,*zo,*to;
    if(!PyArg_ParseTuple(args,"dOOOOOdi",&w,&bo,&do_,&mo,&zo,&to,&ne,&model))return NULL;
    PyArrayObject *b=(PyArrayObject*)PyArray_FROM_OTF(bo,NPY_DOUBLE,NPY_ARRAY_CARRAY_RO),*d=(PyArrayObject*)PyArray_FROM_OTF(do_,NPY_DOUBLE,NPY_ARRAY_CARRAY_RO),*m=(PyArrayObject*)PyArray_FROM_OTF(mo,NPY_DOUBLE,NPY_ARRAY_CARRAY_RO),*z=(PyArrayObject*)PyArray_FROM_OTF(zo,NPY_INT32,NPY_ARRAY_CARRAY_RO),*t=(PyArrayObject*)PyArray_FROM_OTF(to,NPY_DOUBLE,NPY_ARRAY_CARRAY_RO);
    if(!b||!d||!m||!z||!t)goto fail;
    int n=(int)PyArray_SIZE(d);
    if(PyArray_SIZE(b)!=3||PyArray_SIZE(m)!=n||PyArray_SIZE(z)!=n||PyArray_SIZE(t)<n+1) {
        PyErr_SetString(PyExc_ValueError,"invalid cold-plasma array shapes");
        goto fail;
    }
    npy_intp shape[2]= {
        3,3
    }
    ;
    PyArrayObject *o=(PyArrayObject*)PyArray_ZEROS(2,shape,NPY_COMPLEX128,0);
    int rc=petram_cold_std_v1(w,PyArray_DATA(b),PyArray_DATA(d),PyArray_DATA(m),PyArray_DATA(z),PyArray_DATA(t),n,ne,model,PyArray_DATA(o));
    Py_DECREF(b);
    Py_DECREF(d);
    Py_DECREF(m);
    Py_DECREF(z);
    Py_DECREF(t);
    if(rc) {
        Py_DECREF(o);
        PyErr_SetString(PyExc_ValueError,"cold native ABI failure");
        return NULL;
    }
    return (PyObject*)o;
    fail: Py_XDECREF(b);
    Py_XDECREF(d);
    Py_XDECREF(m);
    Py_XDECREF(z);
    Py_XDECREF(t);
    return NULL;
}
static PyObject *address(PyObject *s, PyObject *a) {
    return PyLong_FromVoidPtr((void*)petram_cold_std_v1);
}
static PyMethodDef methods[]= {
    {
        "epsilonr_pl_cold_std",cold_std_py,METH_VARARGS,"Evaluate cold dielectric tensor."
    }
    , {
        "_cold_std_address",address,METH_NOARGS,"Return v1 C ABI address."
    }
    , {
        NULL,NULL,0,NULL
    }
}
;
static struct PyModuleDef module= {
    PyModuleDef_HEAD_INIT,"_rf_dispersion_coldplasma_ext",NULL,-1,methods
}
;
PyMODINIT_FUNC PyInit__rf_dispersion_coldplasma_ext(void) {
    import_array();
    return PyModule_Create(&module);
}
