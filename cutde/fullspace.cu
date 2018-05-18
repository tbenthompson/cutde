<%
def comp(d):
    return ['x', 'y', 'z', 'a', 'b', 'c'][d]
%>
${cluda_preamble}

#define Real ${float_type}

typedef struct Real2 {
    Real x;
    Real y;
} Real2;

typedef struct Real3 {
    Real x;
    Real y;
    Real z;
} Real3;

typedef struct Real6 {
    Real x;
    Real y;
    Real z;
    Real a;
    Real b;
    Real c;
} Real6;

<%def name="binop(dim, name, op, b_scalar = False)">
<%
b_type = 'Real' if b_scalar else 'Real' + str(dim)
%>
WITHIN_KERNEL Real${dim} ${name}${dim}(Real${dim} a, ${b_type} b) {
    Real${dim} out;
    % for d in range(dim):
        <%
            if b_scalar:
                b_str = 'b';
            else:
                b_str = 'b.' + comp(d);
        %>
        out.${comp(d)} = a.${comp(d)} ${op} ${b_str};
    % endfor
    return out;
}
</%def>

% for dim in [2,3,6]:
${binop(dim, 'add','+')}
${binop(dim, 'sub','-')}
${binop(dim, 'mul','*')}
${binop(dim, 'div','/')}

${binop(dim, 'add_scalar','+',b_scalar = True)}
${binop(dim, 'sub_scalar','-',b_scalar = True)}
${binop(dim, 'mul_scalar','*',b_scalar = True)}
${binop(dim, 'div_scalar','/',b_scalar = True)}

WITHIN_KERNEL Real sum${dim}(Real${dim} x) {
    Real out = 0.0;
    % for d in range(dim):
        out += x.${comp(d)};
    % endfor
    return out;
}

WITHIN_KERNEL Real dot${dim}(Real${dim} x, Real${dim} y) {
    return sum${dim}(mul${dim}(x, y));
}

WITHIN_KERNEL Real length${dim}(Real${dim} x) {
    return sqrt(dot${dim}(x, x));
}

WITHIN_KERNEL Real${dim} negate${dim}(Real${dim} x) {
    return mul_scalar${dim}(x, -1);
}

WITHIN_KERNEL Real${dim} normalize${dim}(Real${dim} x) {
    return div_scalar${dim}(x, length${dim}(x));
}

WITHIN_KERNEL Real${dim} make${dim}(
% for d in range(dim):
    Real ${comp(d)}
    % if d != dim - 1:
        ,
    % endif
% endfor
) {
    Real${dim} out;
    % for d in range(dim):
        out.${comp(d)} = ${comp(d)};
    % endfor
    return out;
}

% endfor


WITHIN_KERNEL Real2 transform2(Real2 a, Real2 b, Real2 v) {
    Real2 out;
    out.x = dot2(a,v);
    out.y = dot2(b,v);
    return out;
}

WITHIN_KERNEL Real2 inv_transform2(Real2 a, Real2 b, Real2 v) {
    Real2 out;
    out.x = a.x * v.x + b.x * v.y;
    out.y = a.y * v.x + b.y * v.y;
    return out;
}

WITHIN_KERNEL Real3 transform3(Real3 a, Real3 b, Real3 c, Real3 v) {
    Real3 out;
    out.x = dot3(a,v);
    out.y = dot3(b,v);
    out.z = dot3(c,v);
    return out;
}

WITHIN_KERNEL Real3 inv_transform3(Real3 a, Real3 b, Real3 c, Real3 v) {
    Real3 out;
    out.x = a.x * v.x + b.x * v.y + c.x * v.z;
    out.y = a.y * v.x + b.y * v.y + c.y * v.z;
    out.z = a.z * v.x + b.z * v.y + c.z * v.z;
    return out;
}

WITHIN_KERNEL Real3 cross3(Real3 x, Real3 y) {
    Real3 out;
    out.x = x.y * y.z - x.z * y.y;
    out.y = x.z * y.x - x.x * y.z;
    out.z = x.x * y.y - x.y * y.x;
    return out;
}

WITHIN_KERNEL Real6 tensor_transform3(Real3 a, Real3 b, Real3 c, Real6 tensor) {
    Real A[9];
    A[0] = a.x; A[1] = a.y; A[2] = a.z;
    A[3] = b.x; A[4] = b.y; A[5] = b.z;
    A[6] = c.x; A[7] = c.y; A[8] = c.z;
    Real6 out;
    out.x = A[0]*A[0]*tensor.x+2*A[0]*A[3]*tensor.a+2*A[0]*A[6]*tensor.b+
        2*A[3]*A[6]*tensor.c+ A[3]*A[3]*tensor.y+A[6]*A[6]*tensor.z;
    out.y = A[1]*A[1]*tensor.x+2*A[1]*A[4]*tensor.a+2*A[1]*A[7]*tensor.b+
        2*A[4]*A[7]*tensor.c+A[4]*A[4]*tensor.y+A[7]*A[7]*tensor.z;
    out.z = A[2]*A[2]*tensor.x+2*A[2]*A[5]*tensor.a+2*A[2]*A[8]*tensor.b+
        2*A[5]*A[8]*tensor.c+A[5]*A[5]*tensor.y+A[8]*A[8]*tensor.z;
    out.a = A[0]*A[1]*tensor.x+(A[0]*A[4]+A[1]*A[3])*tensor.a+(A[0]*A[7]+
        A[1]*A[6])*tensor.b+(A[7]*A[3]+A[6]*A[4])*tensor.c+A[4]*A[3]*tensor.y+
        A[6]*A[7]*tensor.z;
    out.b = A[0]*A[2]*tensor.x+(A[0]*A[5]+A[2]*A[3])*tensor.a+(A[0]*A[8]+
        A[2]*A[6])*tensor.b+(A[8]*A[3]+A[6]*A[5])*tensor.c+A[5]*A[3]*tensor.y+
        A[6]*A[8]*tensor.z;
    out.c = A[1]*A[2]*tensor.x+(A[2]*A[4]+A[1]*A[5])*tensor.a+(A[2]*A[7]+
        A[1]*A[8])*tensor.b+(A[7]*A[5]+A[8]*A[4])*tensor.c+A[4]*A[5]*tensor.y+
        A[7]*A[8]*tensor.z;
    return out;
}

WITHIN_KERNEL void print_vec(Real3 x) {
    printf("%f, %f, %f\n", x.x, x.y, x.z);
}

WITHIN_KERNEL int trimodefinder(Real3 obs, Real3 tri0, Real3 tri1, Real3 tri2) {
    // trimodefinder calculates the normalized barycentric coordinates of
    // the points with respect to the TD vertices and specifies the appropriate
    // artefact-free configuration of the angular dislocations for the
    // calculations. The input matrices x, y and z share the same size and
    // correspond to the y, z and x coordinates in the TDCS, respectively. p1,
    // p2 and p3 are two-component matrices representing the y and z coordinates
    // of the TD vertices in the TDCS, respectively.
    // The components of the output (trimode) corresponding to each calculation
    // points, are 1 for the first configuration, -1 for the second
    // configuration and 0 for the calculation point that lie on the TD sides.

    Real a = ((tri1.z-tri2.z)*(obs.y-tri2.y)+(tri2.y-tri1.y)*(obs.z-tri2.z))/
        ((tri1.z-tri2.z)*(tri0.y-tri2.y)+(tri2.y-tri1.y)*(tri0.z-tri2.z));
    Real b = ((tri2.z-tri0.z)*(obs.y-tri2.y)+(tri0.y-tri2.y)*(obs.z-tri2.z))/
        ((tri1.z-tri2.z)*(tri0.y-tri2.y)+(tri2.y-tri1.y)*(tri0.z-tri2.z));
    Real c = 1-a-b;

    int result = 1;
    if ((a<=0 && b>c && c>a) ||
            (b<=0 && c>a && a>b) ||
            (c<=0 && a>b && b>c)) {
        result = -1;
    }

    if ((a==0 && b>=0 && c>=0) ||
            (a>=0 && b==0 && c>=0) ||
            (a>=0 && b>=0 && c==0)) {
        result = 0;
    }
    if (result == 0 && obs.x != 0) {
        result = 1;
    }

    return result;
}

WITHIN_KERNEL Real3 AngDisDisp(Real x, Real y, Real z, Real alpha, Real bx, Real by, Real bz, Real nu) {

    Real cosA = cos(alpha);
    Real sinA = sin(alpha);
    Real eta = y*cosA-z*sinA;
    Real zeta = y*sinA+z*cosA;
    Real r = sqrt(x * x + y * y + z * z);

    Real ux = bx/8/M_PI/(1-nu)*(x*y/r/(r-z)-x*eta/r/(r-zeta));
    Real vx = bx/8/M_PI/(1-nu)*(eta*sinA/(r-zeta)-y*eta/r/(r-zeta)+\
        y*y/r/(r-z)+(1-2*nu)*(cosA*log(r-zeta)-log(r-z)));
    Real wx = bx/8/M_PI/(1-nu)*(eta*cosA/(r-zeta)-y/r-eta*z/r/(r-zeta)-\
        (1-2*nu)*sinA*log(r-zeta));

    Real uy = by/8/M_PI/(1-nu)*(x*x*cosA/r/(r-zeta)-x*x/r/(r-z)-\
         (1-2*nu)*(cosA*log(r-zeta)-log(r-z)));
    Real vy = by*x/8/M_PI/(1-nu)*(y*cosA/r/(r-zeta)-sinA*cosA/(r-zeta)-y/r/(r-z));
    Real wy = by*x/8/M_PI/(1-nu)*(z*cosA/r/(r-zeta)-cosA*cosA/(r-zeta)+1/r);

    Real uz = bz*sinA/8/M_PI/(1-nu)*((1-2*nu)*log(r-zeta)-x*x/r/(r-zeta));
    Real vz = bz*x*sinA/8/M_PI/(1-nu)*(sinA/(r-zeta)-y/r/(r-zeta));
    Real wz = bz*x*sinA/8/M_PI/(1-nu)*(cosA/(r-zeta)-z/r/(r-zeta));
    return make3(ux+uy+uz, vx+vy+vz, wx+wy+wz);
}

WITHIN_KERNEL Real3 TDSetupD(Real3 obs, Real alpha, Real3 slip, Real nu, Real3 TriVertex, Real3 SideVec) {
    // TDSetupD transforms coordinates of the calculation points as well as
    // slip vector components from ADCS into TDCS. It then calculates the
    // displacements in ADCS and transforms them into TDCS.

    Real2 A1 = make2(SideVec.z, -SideVec.y);
    Real2 A2 = make2(SideVec.y, SideVec.z);
    Real2 r1 = transform2(A1, A2, make2(obs.y - TriVertex.y, obs.z - TriVertex.z));
    Real y1 = r1.x;
    Real z1 = r1.y;

    Real2 r2 = transform2(A1, A2, make2(slip.y, slip.z));
    Real by1 = r2.x;
    Real bz1 = r2.y;

    Real3 uvw = AngDisDisp(obs.x, y1, z1, -M_PI + alpha, slip.x, by1, bz1, nu);

    Real2 r3 = inv_transform2(A1, A2, make2(uvw.y, uvw.z));
    Real v = r3.x;
    Real w = r3.y;
    return make3(uvw.x, v, w);
}

WITHIN_KERNEL Real6 AngDisStrain(Real x, Real y, Real z, Real alpha, Real bx, Real by, Real bz, Real nu) {
    // AngDisStrain calculates the strains associated with an angular 
    // dislocation in an elastic full-space.

    Real cosA = cos(alpha);
    Real sinA = sin(alpha);
    Real eta = y*cosA-z*sinA;
    Real zeta = y*sinA+z*cosA;

    Real x2 = x*x;
    Real y2 = y*y;
    Real z2 = z*z;
    Real r2 = x2+y2+z2;
    Real r = sqrt(r2);
    Real r3 = r*r2;
    Real rz = r*(r-z);
    Real rmz = (r-z);
    Real r2z2 = r2*rmz*rmz;
    Real r3z = r3*rmz;

    Real W = zeta-r;
    Real W2 = W*W;
    Real Wr = W*r;
    Real W2r = W2*r;
    Real Wr3 = W*r3;
    Real W2r2 = W2*r2;

    Real C = (r*cosA-z)/Wr;
    Real S = (r*sinA-y)/Wr;

    // Partial derivatives of the Burgers' function
    Real rFi_rx = (eta/r/(r-zeta)-y/r/(r-z))/4/M_PI;
    Real rFi_ry = (x/r/(r-z)-cosA*x/r/(r-zeta))/4/M_PI;
    Real rFi_rz = (sinA*x/r/(r-zeta))/4/M_PI;

    Real6 out;
    out.x = bx*(rFi_rx)+
        bx/8/M_PI/(1-nu)*(eta/Wr+eta*x2/W2r2-eta*x2/Wr3+y/rz-
        x2*y/r2z2-x2*y/r3z)-
        by*x/8/M_PI/(1-nu)*(((2*nu+1)/Wr+x2/W2r2-x2/Wr3)*cosA+
        (2*nu+1)/rz-x2/r2z2-x2/r3z)+
        bz*x*sinA/8/M_PI/(1-nu)*((2*nu+1)/Wr+x2/W2r2-x2/Wr3);

    out.y = by*(rFi_ry)+
        bx/8/M_PI/(1-nu)*((1/Wr+S*S-y2/Wr3)*eta+(2*nu+1)*y/rz-y*y*y/r2z2-
        y*y*y/r3z-2*nu*cosA*S)-
        by*x/8/M_PI/(1-nu)*(1/rz-y2/r2z2-y2/r3z+
        (1/Wr+S*S-y2/Wr3)*cosA)+
        bz*x*sinA/8/M_PI/(1-nu)*(1/Wr+S*S-y2/Wr3);

    out.z = bz*(rFi_rz)+
        bx/8/M_PI/(1-nu)*(eta/W/r+eta*C*C-eta*z2/Wr3+y*z/r3+
        2*nu*sinA*C)-
        by*x/8/M_PI/(1-nu)*((1/Wr+C*C-z2/Wr3)*cosA+z/r3)+
        bz*x*sinA/8/M_PI/(1-nu)*(1/Wr+C*C-z2/Wr3);

    out.a = bx*(rFi_ry)/2+by*(rFi_rx)/2-
        bx/8/M_PI/(1-nu)*(x*y2/r2z2-nu*x/rz+x*y2/r3z-nu*x*cosA/Wr+
        eta*x*S/Wr+eta*x*y/Wr3)+
        by/8/M_PI/(1-nu)*(x2*y/r2z2-nu*y/rz+x2*y/r3z+nu*cosA*S+
        x2*y*cosA/Wr3+x2*cosA*S/Wr)-
        bz*sinA/8/M_PI/(1-nu)*(nu*S+x2*S/Wr+x2*y/Wr3);

    out.b = bx*(rFi_rz)/2+bz*(rFi_rx)/2-
        bx/8/M_PI/(1-nu)*(-x*y/r3+nu*x*sinA/Wr+eta*x*C/Wr+
        eta*x*z/Wr3)+
        by/8/M_PI/(1-nu)*(-x2/r3+nu/r+nu*cosA*C+x2*z*cosA/Wr3+
        x2*cosA*C/Wr)-
        bz*sinA/8/M_PI/(1-nu)*(nu*C+x2*C/Wr+x2*z/Wr3);

    out.c = by*(rFi_rz)/2+bz*(rFi_ry)/2+
        bx/8/M_PI/(1-nu)*(y2/r3-nu/r-nu*cosA*C+nu*sinA*S+eta*sinA*cosA/W2-
        eta*(y*cosA+z*sinA)/W2r+eta*y*z/W2r2-eta*y*z/Wr3)-
        by*x/8/M_PI/(1-nu)*(y/r3+sinA*cosA*cosA/W2-cosA*(y*cosA+z*sinA)/
        W2r+y*z*cosA/W2r2-y*z*cosA/Wr3)-
        bz*x*sinA/8/M_PI/(1-nu)*(y*z/Wr3-sinA*cosA/W2+(y*cosA+z*sinA)/
        W2r-y*z/W2r2);
    return out;
}


WITHIN_KERNEL Real6 TDSetupS(Real3 obs, Real alpha, Real3 slip, Real nu,
    Real3 TriVertex, Real3 SideVec) 
{
    // TDSetupS transforms coordinates of the calculation points as well as 
    // slip vector components from ADCS into TDCS. It then calculates the 
    // strains in ADCS and transforms them into TDCS.

    // Transformation matrix
    Real2 A1 = make2(SideVec.z, -SideVec.y);
    Real2 A2 = make2(SideVec.y, SideVec.z);

    // Transform coordinates of the calculation points from TDCS into ADCS
    Real2 r1 = transform2(A1, A2, make2(obs.y - TriVertex.y, obs.z - TriVertex.z));
    Real y1 = r1.x;
    Real z1 = r1.y;

    // Transform the in-plane slip vector components from TDCS into ADCS
    Real2 r2 = transform2(A1, A2, make2(slip.y, slip.z));
    Real by1 = r2.x;
    Real bz1 = r2.y;

    // Calculate strains associated with an angular dislocation in ADCS
    Real6 out_adcs = AngDisStrain(obs.x,y1,z1,-M_PI+alpha,slip.x,by1,bz1,nu);

    // Transform strains from ADCS into TDCS
    Real3 B0 = make3(1.0, 0.0, 0.0);
    Real3 B1 = make3(0.0, A1.x, A1.y);
    Real3 B2 = make3(0.0, A2.x, A2.y);
    return tensor_transform3(B0, B1, B2, out_adcs);
}

<%def name="disp()">
    Real3 out;
    if (mode == 1) {
        // Calculate first angular dislocation contribution
        Real3 r1Tp = TDSetupD(transformed_obs,A,slip,nu,transformed_tri0, negate3(e13));
        // Calculate second angular dislocation contribution
        Real3 r2Tp = TDSetupD(transformed_obs,B,slip,nu,transformed_tri1, e12);
        // Calculate third angular dislocation contribution
        Real3 r3Tp = TDSetupD(transformed_obs,C,slip,nu,transformed_tri2, e23);
        out = add3(add3(r1Tp, r2Tp), r3Tp);
    } else if (mode == -1) {
        // Calculate first angular dislocation contribution
        Real3 r1Tn = TDSetupD(transformed_obs,A,slip,nu,transformed_tri0,e13);
        // Calculate second angular dislocation contribution
        Real3 r2Tn = TDSetupD(transformed_obs,B,slip,nu,transformed_tri1,negate3(e12));
        // Calculate third angular dislocation contribution
        Real3 r3Tn = TDSetupD(transformed_obs,C,slip,nu,transformed_tri2,negate3(e23));
        out = add3(add3(r1Tn, r2Tn), r3Tn);
    } else {
        out = make3(NAN, NAN, NAN);
    }

    Real3 a = make3(
        -transformed_obs.x,
        transformed_tri0.y - transformed_obs.y,
        transformed_tri0.z - transformed_obs.z
    );
    Real3 b = negate3(transformed_obs);
    Real3 c = make3(
        -transformed_obs.x,
        transformed_tri2.y - transformed_obs.y,
        transformed_tri2.z - transformed_obs.z
    );
    Real na = length3(a);
    Real nb = length3(b);
    Real nc = length3(c);

    Real FiN = (a.x*(b.y*c.z-b.z*c.y)- \
           a.y*(b.x*c.z-b.z*c.x)+ \
           a.z*(b.x*c.y-b.y*c.x));
    Real FiD = (na*nb*nc+dot3(a,b)*nc+dot3(a,c)*nb+dot3(b,c)*na);
    Real Fi = -2*atan2(FiN,FiD)/4/M_PI;

    // Calculate the complete displacement vector components in TDCS
    out = add3(out, mul_scalar3(slip,Fi));

    // Transform the complete displacement vector components from TDCS into EFCS
    Real3 final = inv_transform3(Vnorm, Vstrike, Vdip, out);
    %for d in range(3):
        results[out_idx * 3 + ${d}] = final.${comp(d)};
    %endfor
</%def>

<%def name="strain()">
    Real6 out;
    if (mode == 1) {
        // Calculate first angular dislocation contribution
        Real6 comp1 = TDSetupS(transformed_obs,A,slip,nu,transformed_tri0, negate3(e13));
        // Calculate second angular dislocation contribution
        Real6 comp2 = TDSetupS(transformed_obs,B,slip,nu,transformed_tri1, e12);
        // Calculate third angular dislocation contribution
        Real6 comp3 = TDSetupS(transformed_obs,C,slip,nu,transformed_tri2, e23);
        out = add6(add6(comp1, comp2), comp3);
    } else if (mode == -1) {
        // Calculate first angular dislocation contribution
        Real6 comp1 = TDSetupS(transformed_obs,A,slip,nu,transformed_tri0,e13);
        // Calculate second angular dislocation contribution
        Real6 comp2 = TDSetupS(transformed_obs,B,slip,nu,transformed_tri1,negate3(e12));
        // Calculate third angular dislocation contribution
        Real6 comp3 = TDSetupS(transformed_obs,C,slip,nu,transformed_tri2,negate3(e23));
        out = add6(add6(comp1, comp2), comp3);
    } else {
        out = make6(NAN, NAN, NAN, NAN, NAN, NAN);
    }


    Real6 final = tensor_transform3(Vnorm, Vstrike, Vdip, out);

    /*Real6 final = tensor_transform3(*/
    /*    make3(Vnorm.x, Vstrike.x, Vdip.x),*/
    /*    make3(Vnorm.y, Vstrike.y, Vdip.y),*/
    /*    make3(Vnorm.z, Vstrike.z, Vdip.z),*/
    /*    out*/
    /*);*/

    %for d in range(6):
        results[out_idx * 6 + ${d}] = final.${comp(d)};
    %endfor
</%def>

<%def name="setup_tde()">
    Real3 Vnorm = normalize3(cross3(sub3(tri1, tri0), sub3(tri2, tri0)));
    Real3 eY = make3(0.0f, 1.0f, 0.0f);
    Real3 eZ = make3(0.0f,0.0f,1.0f);
    Real3 Vstrike = cross3(eZ, Vnorm);
    if (length3(Vstrike) == 0) {
         Vstrike = mul_scalar3(eY, Vnorm.z);
    }
    Vstrike = normalize3(Vstrike); 
    Real3 Vdip = cross3(Vnorm, Vstrike);

    Real3 transformed_obs = transform3(Vnorm, Vstrike, Vdip, sub3(obs, tri1));
    Real3 transformed_tri0 = transform3(Vnorm, Vstrike, Vdip, sub3(tri0, tri1));
    Real3 transformed_tri1 = make3(0.0f, 0.0f, 0.0f);
    Real3 transformed_tri2 = transform3(Vnorm, Vstrike, Vdip, sub3(tri2, tri1));

    Real3 e12 = normalize3(sub3(transformed_tri1, transformed_tri0));
    Real3 e13 = normalize3(sub3(transformed_tri2, transformed_tri0));
    Real3 e23 = normalize3(sub3(transformed_tri2, transformed_tri1));

    Real A = acos(dot3(e12, e13));
    Real B = acos(dot3(negate3(e12), e23));
    Real C = acos(dot3(e23, e13));

    int mode = trimodefinder(
        transformed_obs,
        transformed_tri0, transformed_tri1, transformed_tri2
    );
</%def>

<%def name="tde(name, evaluator)">
KERNEL
void ${name}_fullspace(GLOBAL_MEM Real* results, int n_pairs, 
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* tris,
    GLOBAL_MEM Real* slips, Real nu)
{
    int i = get_global_id(0);
    int out_idx = i;
    if (i >= n_pairs) {
        return;
    }
    Real3 obs;
    % for d1 in range(3):
        Real3 tri${d1};
        obs.${comp(d1)} = obs_pts[i * 3 + ${d1}];
        % for d2 in range(3):
            tri${d1}.${comp(d2)} = tris[i * 9 + ${d1} * 3 + ${d2}];
        % endfor
    % endfor

    Real3 slip = make3(
        slips[i * 3 + 2],
        slips[i * 3 + 0],
        slips[i * 3 + 1]
    );

    ${setup_tde()}

    ${evaluator()}
}
</%def>

<%def name="tde_all_pairs(name, evaluator)">
KERNEL
void ${name}_fullspace_all_pairs(GLOBAL_MEM Real* results, 
    int n_obs, int n_src,
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* tris,
    GLOBAL_MEM Real* slips, Real nu)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int out_idx = i * n_src + j;

    //TODO: This would probably be a bit faster with some shared memory stuff.
    //TODO: Also, cache some results about each triangle.
    if (i >= n_obs) {
        return;
    }

    if (j >= n_src) {
        return;
    }

    Real3 obs;
    % for d1 in range(3):
        Real3 tri${d1};
        obs.${comp(d1)} = obs_pts[i * 3 + ${d1}];
        % for d2 in range(3):
            tri${d1}.${comp(d2)} = tris[j * 9 + ${d1} * 3 + ${d2}];
        % endfor
    % endfor

    Real3 slip = make3(
        slips[j * 3 + 2],
        slips[j * 3 + 0],
        slips[j * 3 + 1]
    );

    ${setup_tde()}

    ${evaluator()}
}
</%def>

${tde("disp", disp)}
${tde("strain", strain)}
${tde_all_pairs("disp", disp)}
${tde_all_pairs("strain", strain)}
