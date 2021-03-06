<%namespace module="cutde.mako_helpers" import="*"/>
<%namespace name="common" file="common.cu"/>

${common.defs()}

<%def name="matrix(name, evaluator, vec_dim)">
KERNEL
void matrix_${name}(GLOBAL_MEM Real* results, 
    int n_obs, int n_src,
    GLOBAL_MEM Real* obs_pts, GLOBAL_MEM Real* tris,
    Real nu)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

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

    % for d_src in range(3):
    {
        Real3 slip = make3(0.0, 0.0, 0.0);
        slip.${comp([1,2,0][d_src])} = 1.0;

        ${common.setup_tde()}

        ${evaluator()}

        %for d_obs in range(vec_dim):
        {
            int idx = ((i * ${vec_dim} + ${d_obs}) * n_src + j) * 3 + ${d_src};
            results[idx] = final.${comp(d_obs)};
        }
        %endfor
    }
    % endfor
}
</%def>

${matrix("disp", common.disp, 3)}
${matrix("strain", common.strain, 6)}
