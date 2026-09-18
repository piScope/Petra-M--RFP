#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
/* Native ABI v1.  Output is a caller-owned row-major complex128[3][3]. */
#define Q0 1.60217662e-19
#define DA 1.66053906660e-27
#define ME 9.1093837100818e-31
#define MP 1.67262192369e-27
#define CGS_Q 4.80320427e-10
#define CLIGHT 2.99792458e10

double ive(int order, double x);
static void put(double *o,int r,int c,double complex z) {
    o[2*(r*3+c)]=creal(z);
    o[2*(r*3+c)+1]=cimag(z);
}
static double complex get(const double *o,int r,int c) {
    return o[2*(r*3+c)]+I*o[2*(r*3+c)+1];
}
static double complex zfunc(double complex x) {
    struct zfunc_term {
        double complex coefficient;
        double complex pole;
    };
    static const struct zfunc_term terms[] = {
        {1.5286514261986445e-05 + 0.0001338198102574673 * I,
         1.716400180773478 + 1.4077210296411122 * I},
        {-0.04535121664037525 + 0.21770596727006358 * I,
         1.716400180773478 - 1.4077210296411122 * I},
        {8.868862346159413e-05 + 9.518447335868001e-05 * I,
         -1.6671994586793688 + 1.390541660571995 * I},
        {-0.009161315175716632 - 0.2474253047433214 * I,
         -1.6671994586793688 - 1.390541660571995 * I},
        {0.00031896829654753906 - 0.00012225173013488797 * I,
         0.8085401388105352 + 1.4737003544066358 * I},
        {1.625988208680783 - 1.3818455149161408 * I,
         0.8085401388105352 - 1.4737003544066358 * I},
        {-0.0002441894449757977 + 0.0001708062926842305 * I,
         -0.7742194394628614 + 1.4722571763175376 * I},
        {1.5404561025083063 + 1.5857586800922299 * I,
         -0.7742194394628614 - 1.4722571763175376 * I},
        {-0.00020828998612896403 - 0.0003401290008802969 * I,
         0.014033476202404027 + 1.496503032385095 * I},
        {-4.112237248224732 - 0.17403327041934671 * I,
         0.014033476202404027 - 1.496503032385095 * I},
    };
    double complex value;

    if (cabs(x) > 10.0) {
        value = -1.0 / x - 1.0 / (2.0 * cpow(x, 3)) -
                3.0 / (4.0 * cpow(x, 5)) - 15.0 / (8.0 * cpow(x, 7)) -
                105.0 / (16.0 * cpow(x, 9));
    } else {
        value = -1.8545998915445172e-05 - 1.5661929389884643e-05 * I;
        for (int k = 0; k < 10; ++k) {
            value += terms[k].coefficient / (x - terms[k].pole);
        }
    }
    return creal(value) + I * sqrt(M_PI) * exp(-creal(x) * creal(x));
}
static double om(double f) {
    return 2.*M_PI*f;
}
static double wce(double b,double f) {
    return -CGS_Q*b*1e4/(ME*1000.*CLIGHT)/om(f);
}
static double wci(double b,double f,double a,double z) {
    return CGS_Q*z*b*1e4/(a*MP*1000.*CLIGHT)/om(f);
}
static double wpesq(double ne,double f) {
    return 4.*M_PI*ne*1e-6*CGS_Q*CGS_Q/(ME*1000.)/(om(f)*om(f));
}
static double wpisq(double ni,double a,double z,double f) {
    return 4.*M_PI*ni*1e-6*(CGS_Q*z)*(CGS_Q*z)/(a*MP*1000.)/(om(f)*om(f));
}
static double vte(double t) {
    return sqrt(2.*Q0*1e10*t/(ME*1000.))/CLIGHT;
}
static double vti(double t,double a) {
    return sqrt(2.*Q0*1e10*t/(a*MP*1000.))/CLIGHT;
}
static void chi_el(double np,double nz,double ne,double te,double b,double f,int h,double complex *q) {
    double wc=wce(b,f),lam=np*np*vte(te)*vte(te)/(2*wc*wc), im=ive(h,lam)-(ive(h-1,lam)+ive(h+1,lam))/2., nin=(ive(h-1,lam)-ive(h+1,lam))/2., vt=vte(te), ze=(1.-h*wc)/(nz*vt);
    double complex zz=zfunc(ze);
    if(nz<0)zz=creal(zz)-I*cimag(zz);
    double complex A=zz/(om(f)*nz*vt)*wpesq(ne,f)*om(f),B=CLIGHT/(om(f)*nz)*(1.+ze*zz)*wpesq(ne,f)*om(f);
    q[0]=h*nin*A;
    q[1]=-I*h*im*A;
    q[2]=q[0]+2*lam*im*A;
    q[3]=np/wc/CLIGHT*nin*B;
    q[4]=I*np/wc/CLIGHT*im*B;
    q[5]=2./CLIGHT/nz/(vt*vt)*(1.-h*wc)*ive(h,lam)*B;
}
static void chi_i(double np,double nz,double ni,double a,double zc,double ti,double b,double f,int h,double complex *q) {
    double wc=wci(b,f,a,zc),vt=vti(ti,a),lam=np*np*vt*vt/(2*wc*wc),im=ive(h,lam)-(ive(h-1,lam)+ive(h+1,lam))/2.,nin=(ive(h-1,lam)-ive(h+1,lam))/2.,ze=(1.-h*wc)/(nz*vt);
    double complex z=zfunc(ze);
    if(nz<0)z=creal(z)-I*cimag(z);
    double complex A=z/(om(f)*nz*vt)*wpisq(ni,a,zc,f)*om(f),B=CLIGHT/(om(f)*nz)*(1.+ze*z)*wpisq(ni,a,zc,f)*om(f);
    q[0]=h*nin*A;
    q[1]=-I*h*im*A;
    q[2]=q[0]+2*lam*im*A;
    q[3]=np/wc/CLIGHT*nin*B;
    q[4]=I*np/wc/CLIGHT*im*B;
    q[5]=2./CLIGHT/nz/(vt*vt)*(1.-h*wc)*ive(h,lam)*B;
}
static void add(double *o,double complex *q,const int32_t*t,double nuc,double w) {
    double complex r[6]= {
        0
    }
    ,a[9],m[9];
    if(t[0])r[0]=q[0];
    if(t[1])r[1]=q[1];
    if(t[2])r[5]=q[5];
    if(t[3]&&t[0])r[2]=q[2];
    else if(!t[3]&&t[0])r[2]=q[0];
    else if(t[3])r[2]=q[2]-q[0];
    if(t[4])r[3]=q[3];
    if(t[5])r[4]=q[4];
    if(!t[6])for(int i=0;i<6;i++)r[i]-=creal(r[i]);
    if(!t[7])for(int i=0;i<6;i++)r[i]=creal(r[i]);
    m[0]=r[0];
    m[1]=r[1];
    m[2]=r[3];
    m[3]=-r[1];
    m[4]=r[2];
    m[5]=r[4];
    m[6]=r[3];
    m[7]=-r[4];
    m[8]=r[5];
    for(int i=0;i<9;i++) {
        a[i]=I*(m[i]+conj(m[(i%3)*3+i/3]))*.5*nuc/w;
        put(o,i/3,i%3,get(o,i/3,i%3)+m[i]+a[i]);
    }
}
int petram_hot_std_v1(double w,const double*b,const double*temps,const double*dens,const double*masses,const int32_t*z,double te,double ne,double npara,double nperp,int32_t nh,const int32_t*terms,int32_t rows,int32_t eye,const double*nuc,int32_t n,double*out) {
    if(n<0||rows<n+1)return 1;
    for(int i=0;i<18;i++)out[i]=0.;
    if(eye) {
        put(out,0,0,1);
        put(out,1,1,1);
        put(out,2,2,1);
    }
    double bn=sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]),f=w/(2*M_PI);
    double complex q[6];
    if(ne>0) {
        for(int j=0;j<6;j++)q[j]=0.;
        for(int h=-nh;h<=nh;h++) {
            double complex t[6];
            chi_el(nperp,npara,ne,te/1000.,bn,f,h,t);
            for(int j=0;j<6;j++)q[j]+=t[j];
        }
        add(out,q,terms,nuc[0],w);
    }
    for(int k=0;k<n;k++) {
        if(dens[k]<=0)continue;
        for(int j=0;j<6;j++)q[j]=0.;
        for(int h=-nh;h<=nh;h++) {
            double complex t[6];
            chi_i(nperp,npara,dens[k],masses[k]/DA,z[k],temps[k]/1000.,bn,f,h,t);
            for(int j=0;j<6;j++)q[j]+=t[j];
        }
        add(out,q,terms+8*(k+1),nuc[k+1],w);
    }
    return 0;
}
static PyObject* hot_py(PyObject*s,PyObject*args) {
    double w,te,ne,np,nper;
    int nh,eye;
    PyObject *bo,*to,*do_,*mo,*zo,*tro,*no;
    if(!PyArg_ParseTuple(args,"dOOOOOddddiOiO",&w,&bo,&to,&do_,&mo,&zo,&te,&ne,&np,&nper,&nh,&tro,&eye,&no))return NULL;
    PyArrayObject*b=(PyArrayObject*)PyArray_FROM_OTF(bo,NPY_DOUBLE,NPY_ARRAY_CARRAY_RO),*t=(PyArrayObject*)PyArray_FROM_OTF(to,NPY_DOUBLE,NPY_ARRAY_CARRAY_RO),*d=(PyArrayObject*)PyArray_FROM_OTF(do_,NPY_DOUBLE,NPY_ARRAY_CARRAY_RO),*m=(PyArrayObject*)PyArray_FROM_OTF(mo,NPY_DOUBLE,NPY_ARRAY_CARRAY_RO),*z=(PyArrayObject*)PyArray_FROM_OTF(zo,NPY_INT32,NPY_ARRAY_CARRAY_RO),*tr=(PyArrayObject*)PyArray_FROM_OTF(tro,NPY_INT32,NPY_ARRAY_CARRAY_RO),*nu=(PyArrayObject*)PyArray_FROM_OTF(no,NPY_DOUBLE,NPY_ARRAY_CARRAY_RO);
    if(!b||!t||!d||!m||!z||!tr||!nu)goto fail;
    int n=PyArray_SIZE(d);
    if(PyArray_SIZE(b)!=3||PyArray_SIZE(t)!=n||PyArray_SIZE(m)!=n||PyArray_SIZE(z)!=n||PyArray_NDIM(tr)!=2||PyArray_DIM(tr,0)<n+1||PyArray_DIM(tr,1)!=8||PyArray_SIZE(nu)<n+1) {
        PyErr_SetString(PyExc_ValueError,"invalid hot-plasma array shapes");
        goto fail;
    }
    npy_intp sh[2]= {
        3,3
    }
    ;
    PyArrayObject*o=(PyArrayObject*)PyArray_ZEROS(2,sh,NPY_COMPLEX128,0);
    int rc=petram_hot_std_v1(w,PyArray_DATA(b),PyArray_DATA(t),PyArray_DATA(d),PyArray_DATA(m),PyArray_DATA(z),te,ne,np,nper,nh,PyArray_DATA(tr),PyArray_DIM(tr,0),eye,PyArray_DATA(nu),n,PyArray_DATA(o));
    Py_DECREF(b);
    Py_DECREF(t);
    Py_DECREF(d);
    Py_DECREF(m);
    Py_DECREF(z);
    Py_DECREF(tr);
    Py_DECREF(nu);
    if(rc) {
        Py_DECREF(o);
        PyErr_SetString(PyExc_ValueError,"hot native ABI failure");
        return NULL;
    }
    return(PyObject*)o;
    fail:Py_XDECREF(b);
    Py_XDECREF(t);
    Py_XDECREF(d);
    Py_XDECREF(m);
    Py_XDECREF(z);
    Py_XDECREF(tr);
    Py_XDECREF(nu);
    return NULL;
}
static PyObject*address(PyObject*s,PyObject*a) {
    return PyLong_FromVoidPtr((void*)petram_hot_std_v1);
}
static PyObject *zfunc_py(PyObject *self, PyObject *args) {
    Py_complex input;
    double complex output;

    if (!PyArg_ParseTuple(args, "D", &input)) {
        return NULL;
    }
    output = zfunc(input.real + I * input.imag);
    return PyComplex_FromDoubles(creal(output), cimag(output));
}
static PyObject *ive_py(PyObject *self, PyObject *args) {
    int order;
    double x;

    if (!PyArg_ParseTuple(args, "id", &order, &x)) {
        return NULL;
    }
    return PyFloat_FromDouble(ive(order, x));
}
static PyMethodDef methods[]= {
    {
        "epsilonr_pl_hot_std",hot_py,METH_VARARGS,"Evaluate hot dielectric tensor."
    }
    , {
        "_hot_std_address",address,METH_NOARGS,"Return v1 C ABI address."
    }
    , {
        "_zfunc", zfunc_py, METH_VARARGS,
        "Evaluate the native plasma-dispersion approximation."
    }
    , {
        "_ive", ive_py, METH_VARARGS,
        "Evaluate the native exponentially scaled modified Bessel I."
    }
    , {
        NULL,NULL,0,NULL
    }
}
;
static struct PyModuleDef module= {
    PyModuleDef_HEAD_INIT,"_rf_dispersion_lkplasma_ext",NULL,-1,methods
}
;
PyMODINIT_FUNC PyInit__rf_dispersion_lkplasma_ext(void) {
    import_array();
    return PyModule_Create(&module);
}
